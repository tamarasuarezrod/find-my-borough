import { Borough } from '@/types/borough'
import { useQuery } from '@tanstack/react-query'

export const getBoroughBySlug = async (slug: string): Promise<Borough> => {
  const res = await fetch(`http://127.0.0.1:8000/api/boroughs/${slug}/`)
  if (!res.ok) throw new Error('Failed to fetch borough detail')
  return res.json()
}

export const useBoroughBySlug = (slug: string) =>
  useQuery({
    queryKey: ['borough', slug],
    queryFn: () => (slug ? getBoroughBySlug(slug) : Promise.reject('No slug')),
    enabled: !!slug,
  })
