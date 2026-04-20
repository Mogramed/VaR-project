export function isRenderableDatum(value: unknown): boolean {
  if (value == null) {
    return false;
  }
  if (typeof value === "number") {
    return Number.isFinite(value);
  }
  if (Array.isArray(value)) {
    return value.some((item) => isRenderableDatum(item));
  }
  if (typeof value === "object") {
    const pointValue = (value as { value?: unknown }).value;
    return pointValue === undefined ? false : isRenderableDatum(pointValue);
  }
  return false;
}

export function countRenderablePoints(option: Record<string, unknown>): number {
  const series = option.series;
  if (!Array.isArray(series)) {
    return 0;
  }

  return series.reduce((count, item) => {
    if (item == null || typeof item !== "object") {
      return count;
    }
    const data = (item as { data?: unknown }).data;
    if (!Array.isArray(data)) {
      return count;
    }
    return count + data.filter((datum) => isRenderableDatum(datum)).length;
  }, 0);
}
