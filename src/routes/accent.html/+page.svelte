<script lang="ts">
	import '$lib/studio/studio.css';
	import '$lib/studio/accent.css';
	import { mountAccent } from '$lib/accent';
	import { onMount } from 'svelte';

	let { data } = $props();
	onMount(mountAccent);
</script>

<svelte:head>
	{@html data.head}
</svelte:head>

{@html data.body}
