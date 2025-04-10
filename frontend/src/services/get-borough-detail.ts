import { api } from '@/lib/axios'
import { Borough } from '@/types/borough'
import { useQuery } from '@tanstack/react-query'

export const getBoroughBySlug = async (slug: string): Promise<Borough> => {
  const res = await api.get(`/boroughs/${slug}/`)
  return res.data
}

export const useBoroughBySlug = (slug: string) =>
  useQuery({
    queryKey: ['borough', slug],
    queryFn: () => (slug ? getBoroughBySlug(slug) : Promise.reject('No slug')),
    enabled: !!slug,
  })
