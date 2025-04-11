import { api } from '@/lib/axios'
import { useQuery, useMutation } from '@tanstack/react-query'

export const useSubmitCommunityRatings = () =>
  useMutation({
    mutationFn: async ({
      borough,
      ratings,
    }: {
      borough: string
      ratings: Record<string, number>
    }) => {
      const res = await api.post('/boroughs/community/submit/', {
        borough,
        ratings: Object.entries(ratings).map(([feature, score]) => ({
          [feature]: score,
        })),
      })
      return res.data
    },
  })

export const useCommunityScores = (slug: string) =>
  useQuery({
    queryKey: ['community-scores', slug],
    queryFn: async () => {
      const res = await api.get(`/boroughs/community/scores/${slug}/`)
      return res.data
    },
    enabled: !!slug,
  })
