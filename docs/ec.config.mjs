// @ts-check
import { defineEcConfig, definePlugin } from '@astrojs/starlight/expressive-code';

const languageLabels = {
	bash: 'Bash',
	c: 'C',
	cpp: 'C++',
	css: 'CSS',
	cu: 'CUDA',
	cuda: 'CUDA',
	diff: 'Diff',
	dockerfile: 'Dockerfile',
	html: 'HTML',
	ini: 'INI',
	js: 'JavaScript',
	json: 'JSON',
	jsonc: 'JSON',
	jsx: 'JSX',
	matlab: 'MATLAB',
	md: 'Markdown',
	mdx: 'MDX',
	powershell: 'PowerShell',
	ps1: 'PowerShell',
	py: 'Python',
	python: 'Python',
	sh: 'Shell',
	shell: 'Shell',
	sql: 'SQL',
	toml: 'TOML',
	ts: 'TypeScript',
	tsx: 'TSX',
	yaml: 'YAML',
	yml: 'YAML',
	zsh: 'Zsh',
};

const unlabeledLanguages = new Set(['', 'ansi', 'plaintext', 'text', 'txt']);

/** Titles untitled multi-line blocks with their language, shown in the frame header. */
const languageTitle = definePlugin({
	name: 'language-title',
	hooks: {
		// Runs after the frames plugin, so titles from file name comments take precedence.
		preprocessCode: ({ codeBlock }) => {
			const { language, props } = codeBlock;
			if (props.title || codeBlock.getLines().length < 2 || unlabeledLanguages.has(language)) return;
			props.title =
				languageLabels[/** @type {keyof typeof languageLabels} */ (language)] ??
				language.charAt(0).toUpperCase() + language.slice(1);
		},
	},
});

export default defineEcConfig({
	themes: ['github-dark-default', 'github-light-default'],
	defaultProps: { frame: 'code' },
	plugins: [languageTitle],
	styleOverrides: {
		borderRadius: '0.75rem',
		borderColor: 'var(--sl-color-hairline-light)',
		codeBackground: 'var(--cunibs-surface-raised)',
		codeFontSize: 'var(--sl-text-code)',
		codeLineHeight: '1.6',
		codePaddingBlock: '1rem',
		codePaddingInline: '1rem',
		uiFontSize: 'var(--sl-text-xs)',
		uiPaddingBlock: '0.5rem',
		uiPaddingInline: '1rem',
		frames: {
			frameBoxShadowCssValue: 'none',
			editorTabBarBackground: 'var(--cunibs-surface-raised)',
			editorTabBarBorderColor: 'var(--sl-color-hairline-light)',
			editorTabBarBorderBottomColor: 'var(--sl-color-hairline-light)',
			editorTabBorderRadius: '0',
			editorTabsMarginInlineStart: '0',
			editorTabsMarginBlockStart: '0',
			editorActiveTabBackground: 'transparent',
			editorActiveTabBorderColor: 'transparent',
			editorActiveTabForeground: 'var(--sl-color-gray-3)',
			editorActiveTabIndicatorHeight: '0px',
			editorActiveTabIndicatorTopColor: 'transparent',
			editorActiveTabIndicatorBottomColor: 'transparent',
			inlineButtonForeground: 'var(--sl-color-gray-3)',
			inlineButtonBackground: 'var(--sl-color-white)',
			inlineButtonBackgroundHoverOrFocusOpacity: '0.08',
			inlineButtonBackgroundActiveOpacity: '0.12',
			inlineButtonBorderOpacity: '0',
		},
	},
});
