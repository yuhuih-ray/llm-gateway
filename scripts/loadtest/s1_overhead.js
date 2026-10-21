import exec from 'k6/execution';
import { Trend } from 'k6/metrics';
import { arrival, baseOptions, completion } from './common.js';

const smoke = __ENV.LOADTEST_SMOKE === '1';
const rates = smoke ? [20] : [100, 200, 400, 800, 1600];
const duration = smoke ? 10 : 60;
const scenarios = {};
const latency = {};
for (let index = 0; index < rates.length; index++) {
  const rate = rates[index];
  const name = `rps_${rate}`;
  scenarios[name] = arrival(rate, `${duration}s`, `${index * duration}s`);
  // Separate trends retain each plateau in the --summary-export output.
  latency[name] = new Trend(`latency_${name}`, true);
}
export const options = { ...baseOptions, scenarios };
export default function () {
  const response = completion();
  latency[exec.scenario.name].add(response.timings.duration);
}
