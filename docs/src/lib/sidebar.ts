import type { StarlightRouteData } from '@astrojs/starlight/route-data';

export type SidebarEntry = StarlightRouteData['sidebar'][number];
export type SidebarGroup = Extract<SidebarEntry, { type: 'group' }>;

function groupHasCurrent(group: SidebarGroup): boolean {
	return group.entries.some((entry) =>
		entry.type === 'link' ? entry.isCurrent : groupHasCurrent(entry)
	);
}

export function currentTrail(entries: SidebarEntry[]): string[] {
	for (const entry of entries) {
		if (entry.type === 'link') {
			if (entry.isCurrent) return [];
			continue;
		}
		if (groupHasCurrent(entry)) return [entry.label, ...currentTrail(entry.entries)];
	}
	return [];
}
