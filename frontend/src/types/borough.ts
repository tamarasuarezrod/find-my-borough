export type Borough = {
  name: string
  slug: string
  image: string
  norm_rent?: number
  norm_crime?: number
  norm_youth?: number
  norm_centrality: number
}

export type Recommendation = {
  borough: string
  score: number
  norm_rent: number
  norm_crime: number
  norm_youth: number
  norm_centrality: number
}

export type UserAnswers = {
  budget_weight: string
  current_situation: string
  safety_weight: string
  stay_duration: string
  centrality_weight: string
  youth_weight: string
}

export type Score = {
  feature: string
  label: string
  score: number
}

export type BoroughScore = {
  borough: string
  ratings: Record<string, number>
}
