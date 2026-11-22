/** Fraction of the viewport, from the top, a heading must pass to become the current section. */
const ACTIVATION_LINE = 0.25;
const TOP_OF_PAGE_PX = 50;

function isScrolledToBottom() {
	return scrollY + innerHeight >= document.documentElement.scrollHeight - 1;
}

function activeHeadingId(headings: HTMLElement[]): string | null {
	if (headings.length === 0) return null;
	const line = innerHeight * ACTIVATION_LINE;
	const atBottom = isScrolledToBottom();

	// Headings near the end of a page can never scroll up to the line, so a heading the reader
	// navigated to stays current while it is on screen.
	const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
	if (target && headings.includes(target)) {
		const { top } = target.getBoundingClientRect();
		if (top >= 0 && (atBottom || top <= line)) return target.id;
	}

	if (scrollY < TOP_OF_PAGE_PX) return headings[0]!.id;
	if (atBottom) return headings.at(-1)!.id;
	const passed = headings.filter((heading) => heading.getBoundingClientRect().top <= line);
	return (passed.at(-1) ?? headings[0]!).id;
}

/** Calls `onChange` with the id of the section being read whenever it changes. */
export function observeActiveHeading(ids: string[], onChange: (id: string | null) => void) {
	const headings = ids.flatMap((id) => document.getElementById(id) ?? []);
	let current: string | null | undefined;
	let frame = 0;

	const update = () => {
		frame = 0;
		const next = activeHeadingId(headings);
		if (next === current) return;
		current = next;
		onChange(next);
	};
	const schedule = () => {
		frame ||= requestAnimationFrame(update);
	};

	addEventListener('scroll', schedule, { passive: true });
	addEventListener('resize', schedule, { passive: true });
	addEventListener('hashchange', schedule);
	update();

	return () => {
		removeEventListener('scroll', schedule);
		removeEventListener('resize', schedule);
		removeEventListener('hashchange', schedule);
		cancelAnimationFrame(frame);
	};
}
