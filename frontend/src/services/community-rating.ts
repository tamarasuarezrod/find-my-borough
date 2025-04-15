import { api } from '@/lib/axios'
import { BoroughScore, Score } from '@/types/borough'
import {
  useMutation,
  UseMutationOptions,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query'
import { ApiError } from 'next/dist/server/api-utils'

export const useSubmitCommunityRatings = (
  props: UseMutationOptions<BoroughScore, ApiError, BoroughScore>,
) => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({ borough, ratings }: BoroughScore) => {
      const res = await api.post('/boroughs/community/submit/', {
        borough,
        ratings: Object.entries(ratings).map(([feature, score]) => ({
          [feature]: score,
        })),
      })
      return res.data
    },
    onSuccess: (data, variables, context) => {
      props?.onSuccess?.(data, variables, context)
      queryClient.invalidateQueries({
        queryKey: ['community-scores', variables.borough],
      })
    },
  })
}

export const useCommunityScores = (slug: string) =>
  useQuery<Score[]>({
    queryKey: ['community-scores', slug],
    queryFn: async () => {
      const res = await api.get(`/boroughs/community/scores/${slug}/`)
      return res.data
    },
    enabled: !!slug,
  })
