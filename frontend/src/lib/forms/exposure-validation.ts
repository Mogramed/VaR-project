export const EXPOSURE_STEP_EUR = "any";
export const EXPOSURE_MIN_EUR = 0;
export const EXPOSURE_VALIDATION_MESSAGE =
  "Exposure must be a positive numeric value. MT5 broker constraints are checked at preview/submit time.";

export function isExposureMagnitudeValid(value: number): boolean {
  return Number.isFinite(value) && value > 0;
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
