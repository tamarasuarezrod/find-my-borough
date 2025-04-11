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

export type Score = {
  feature: string
  label: string
  score: number
}
