// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
	integrations: [
		starlight({
			title: 'cuNIBS Docs',
			logo: { src: './src/assets/logo.svg' },
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/vcubiomag/cunibs' }],
			favicon: '/favicon.svg',
			head: [
				{ tag: 'link', attrs: { rel: 'icon', href: '/favicon.ico', sizes: '32x32' } },
			],
			customCss: ['./src/styles/theme.css'],
			sidebar: [
				{
					label: 'Getting started',
					items: [
						{ label: 'Overview', link: '/' },
						'start/installation',
						'start/first-simulation',
					],
				},
				{
					label: 'Inputs',
					items: ['inputs/head-models', 'inputs/coils', 'inputs/coordinates'],
				},
				{
					label: 'Guides',
					items: [
						'guides/compare-placements',
						'guides/process-a-cohort',
						'guides/region-and-depth',
						'guides/conductivity-uncertainty',
						'guides/optimize-placement',
					],
				},
				{
					label: 'Results and reference',
					items: [
						'results/interpret-results',
						'results/save-and-reuse',
						{
							label: 'Python API',
							items: [
								'reference/simulation',
								'reference/inputs-and-head',
								'reference/results-and-metrics',
								'reference/uncertainty',
								'reference/optimization',
								'reference/fem',
							],
						},
						'reference/methods-and-reproducibility',
					],
				},
				{ label: 'Project', items: ['project/developing', 'project/contact', 'project/citation'] },
			],
			components: {
				Header: './src/components/Header.astro',
				Sidebar: './src/components/Sidebar.astro',
				SocialIcons: './src/components/SocialIcons.astro',
				ThemeSelect: './src/components/ThemeSelect.astro',
				PageTitle: './src/components/PageTitle.astro',
				TableOfContents: './src/components/TableOfContents.astro',
				MobileTableOfContents: './src/components/MobileTableOfContents.astro',
			},
		}),
	],
});
