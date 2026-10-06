import body from '$lib/studio/accent-body.html?raw';
import head from '$lib/studio/accent-head.html?raw';

import type { PageLoad } from './$types';

// Experimental accent page (fork): local only, served with the accent API from server.py.
export const prerender = true;
export const trailingSlash = 'never';
export const load: PageLoad = () => ({ head, body });
