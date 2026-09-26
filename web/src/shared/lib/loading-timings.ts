// Shared loading-state contract: a 140ms show delay plus a 280ms
// minimum-visible window keep background refetches from flashing skeletons.
// Both frontends' useStableLoading implementations consume this.
export const loadingTimings = {
  delayMs: 140,
  minimumVisibleMs: 280,
} as const
