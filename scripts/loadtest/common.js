import http from 'k6/http';
import { check } from 'k6';
import exec from 'k6/execution';
import { SharedArray } from 'k6/data';
import { Counter } from 'k6/metrics';

const keys = new SharedArray('tenant keys', () => JSON.parse(open('./.state/keys.json')));
if (keys.length !== 50) throw new Error('Run setup.sh: exactly 50 tenant keys are required');
const requestsSent = new Counter('requests_sent');
const requestsSuccessful = new Counter('requests_successful');

export const baseOptions = {
  discardResponseBodies: true,
  maxRedirects: 0,
  summaryTrendStats: ['avg', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
  thresholds: {
    checks: ['rate==1'],
    http_req_failed: ['rate==0'],
    dropped_iterations: ['count==0'],
  },
};

export function arrival(rate, duration, startTime = '0s', preAllocatedVUs = Math.max(50, Math.ceil(rate / 4))) {
  // Closed VU loops reduce offered RPS as responses slow down, hiding overload.
  // Open arrival-rate scheduling preserves offered load; dropped iterations are failures.
  return {
    executor: 'constant-arrival-rate', rate, timeUnit: '1s', duration, startTime,
    preAllocatedVUs, maxVUs: Math.max(preAllocatedVUs * 2, rate * 2), gracefulStop: '30s',
  };
}

export function completion() {
  // A scenario-wide iteration index gives round-robin keys across VUs.
  const key = keys[exec.scenario.iterationInTest % keys.length];
  requestsSent.add(1);
  const response = http.post('http://127.0.0.1:18000/v1/chat/completions', JSON.stringify({
    model: 'fake', temperature: 0, messages: [{ role: 'user', content: 'Load-test request' }],
  }), {
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
    timeout: '30s',
    tags: { name: 'chat-completions' },
  });
  // No X-Gateway-Cache header: every request exercises the provider path.
  requestsSuccessful.add(response.status === 200 ? 1 : 0);
  check(response, {
    'status 200': r => r.status === 200,
    'cache bypassed': r => r.headers['X-Cache'] === 'MISS',
  });
  return response;
}
