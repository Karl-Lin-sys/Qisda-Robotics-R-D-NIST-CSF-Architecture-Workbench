import test from 'node:test';
import assert from 'node:assert/strict';
import {
  calculateComplexityMetrics,
  computeBalancedSoftmaxProbabilities,
  evaluateMahalanobisGate,
  auditPayloadForTaiwanGitHubCompliance,
  CONFERENCE_ARCHITECTURES,
  NIST_CSF_FUNCTIONS
} from './roboticsEngine';

test('calculateComplexityMetrics scales deterministically with resolution and point count', () => {
  const small = calculateComplexityMetrics({
    height: 240,
    width: 320,
    numPoints: 10000,
    batchSize: 4,
    numClasses: 20
  });
  const large = calculateComplexityMetrics({
    height: 480,
    width: 640,
    numPoints: 40000,
    batchSize: 4,
    numClasses: 20
  });

  // Doubling both H and W quadruples pixel count and OpenCV O(H*W) operations
  assert.equal(large.opencvWarpOps, small.opencvWarpOps * 4);
  assert.equal(large.opencvPreallocBytes, small.opencvPreallocBytes * 4);
  // Quadrupling N quadruples XLA Mahalanobis O(N * M^2) operations
  assert.equal(large.tfXlaMahalanobisOps, small.tfXlaMahalanobisOps * 4);
});

test('computeBalancedSoftmaxProbabilities prevents majority class dominance during training', () => {
  // Suppose class 0 is common background (9900 samples) and class 1 is rare obstacle (100 samples)
  const result = computeBalancedSoftmaxProbabilities([2.0, 2.0], [9900, 100]);
  // With equal raw logits, naive softmax gives 0.5 / 0.5, while Balanced Softmax shifts training logits
  // by log(prior) so the rare class receives a larger gradient push to overcome the prior gap
  assert.equal(Math.abs(result.naiveProbabilities[0] - 0.5) < 1e-6, true);
  assert.equal(result.probabilities[0] > result.probabilities[1], true);
});

test('evaluateMahalanobisGate accurately classifies nominal vs spoofed sensor innovations', () => {
  const variances = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01];
  const nominal = evaluateMahalanobisGate([0.05, -0.05, 0.02, 0.01, -0.02, 0.01], variances);
  assert.equal(nominal.isNominal, true);

  const spoofed = evaluateMahalanobisGate([1.5, 1.2, 0.8, 0.0, 0.0, 0.0], variances);
  assert.equal(spoofed.isNominal, false);
});

test('auditPayloadForTaiwanGitHubCompliance detects PDPA and Trade Secret markers', () => {
  const cleanCode = 'def forward(x): return x * 2 # Apache-2.0 Clean Room';
  const cleanAudit = auditPayloadForTaiwanGitHubCompliance(cleanCode);
  assert.equal(cleanAudit.isCompliant, true);
  assert.equal(cleanAudit.detectedIssues.length, 0);

  const dirtyCode = 'CONFIDENTIAL applicant A123456789 contact user@example.com';
  const dirtyAudit = auditPayloadForTaiwanGitHubCompliance(dirtyCode);
  assert.equal(dirtyAudit.isCompliant, false);
  assert.equal(dirtyAudit.detectedIssues.length, 3);
  assert.equal(dirtyAudit.sanitizedPreview.includes('[REDACTED_TW_ID]'), true);
});

test('all 6 NIST CSF 2.0 core functions and peer-reviewed architectures are populated', () => {
  assert.equal(NIST_CSF_FUNCTIONS.length, 6);
  assert.equal(CONFERENCE_ARCHITECTURES.length >= 4, true);
});
