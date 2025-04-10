import { Recommendation } from '@/types/borough'
import { useMutation } from '@tanstack/react-query'

export type RecommendationResponse = {
  borough: string
  score: number
  norm_rent: number
  norm_crime: number
  norm_youth: number
  norm_centrality: number
}

export const getRecommendation = async (
  payload: Recommendation,
): Promise<RecommendationResponse[]> => {
  const res = await fetch('http://127.0.0.1:8000/api/recommendations/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  if (!res.ok) {
    throw new Error('Failed to fetch recommendations')
  }

  return res.json()
}

export const useRecommendation = () =>
  useMutation({
    mutationFn: getRecommendation,
  })
