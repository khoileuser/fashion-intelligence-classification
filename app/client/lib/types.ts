export type RankedPrediction = {
  label: string;
  confidence: number;
};

export type Prediction = RankedPrediction & {
  target: string;
  top_k: RankedPrediction[];
  needs_review?: boolean;
  review_reason?: string;
  review_threshold?: number | null;
  confidence_calibrated?: boolean;
};

export type SimilarItem = {
  id: string | number;
  score: number;
  articleType?: string;
  gender?: string;
  season?: string;
  usage?: string;
  productDisplayName?: string;
  [key: string]: unknown;
};

export type AnalysisResponse = {
  predictions: Record<string, Prediction>;
  similar_items: SimilarItem[];
};

export type HealthResponse = {
  status: string;
  models: Record<string, boolean>;
  version: string;
};
