import { check } from 'k6';
import { Trend } from 'k6/metrics';
import { arrival, baseOptions, completion } from './common.js';

const stages = ['auth', 'ratelimit', 'budget', 'cache', 'provider', 'total'];
const timings = {};
for (const stage of stages) timings[stage] = new Trend(`server_${stage}_ms`, true);
export const options = { ...baseOptions, scenarios: { breakdown: arrival(200, '60s') } };

export default function () {
  const response = completion();
  const header = response.headers['Server-Timing'] || '';
  const values = {};
  for (const item of header.split(',')) {
    const match = item.trim().match(/^([a-z]+);dur=([0-9]+(?:\.[0-9]+)?)$/);
    if (match && stages.includes(match[1])) values[match[1]] = Number(match[2]);
  }
  const valid = stages.every(stage => Number.isFinite(values[stage]) && values[stage] >= 0);
  check(valid, { 'all Server-Timing stages present': value => value });
  if (valid) for (const stage of stages) timings[stage].add(values[stage]);
}
