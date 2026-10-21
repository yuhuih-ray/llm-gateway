import { arrival, baseOptions, completion } from './common.js';

// 200 RPS * 2 seconds implies approximately 400 in-flight requests before overhead.
// Reserve 600 VUs so the intended delay does not itself exhaust the generator.
export const options = { ...baseOptions, scenarios: { inflight: arrival(200, '120s', '0s', 600) } };
export default function () { completion(); }
