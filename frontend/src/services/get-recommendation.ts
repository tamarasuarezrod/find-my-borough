import { api } from '@/lib/axios'
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
  const res = await api.post('/recommendations/', payload)
  return res.data
}

export const useRecommendation = () =>
  useMutation({
    mutationFn: getRecommendation,
  })
