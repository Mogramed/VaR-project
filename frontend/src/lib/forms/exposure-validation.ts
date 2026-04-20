export const EXPOSURE_STEP_EUR = 1_000;
export const EXPOSURE_MIN_EUR = 1_000;
export const EXPOSURE_VALIDATION_MESSAGE =
  "Exposure must be at least 1,000 EUR in absolute value and use 1,000 EUR increments.";

const EXPOSURE_EPSILON = 1e-6;

export function isExposureMagnitudeValid(value: number): boolean {
  if (!Number.isFinite(value) || value < EXPOSURE_MIN_EUR) {
    return false;
  }
  const steps = value / EXPOSURE_STEP_EUR;
  return Math.abs(steps - Math.round(steps)) <= EXPOSURE_EPSILON;
}

export function validateExposureMagnitude(input: string): {
  ok: boolean;
  value: number | null;
  error: string | null;
} {
  const parsed = Number(input);
  if (!isExposureMagnitudeValid(parsed)) {
    return {
      ok: false,
      value: null,
      error: EXPOSURE_VALIDATION_MESSAGE,
    };
  }
  return {
    ok: true,
    value: parsed,
    error: null,
  };
}
