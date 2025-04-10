import { Borough } from '@/types/borough'
import { useQuery } from '@tanstack/react-query'

export const getBoroughs = async (): Promise<Borough[]> => {
  const res = await fetch('http://127.0.0.1:8000/api/boroughs/')
  if (!res.ok) throw new Error('Failed to fetch boroughs')
  return res.json()
}

export const useBoroughs = () =>
  useQuery({
    queryKey: ['boroughs'],
    queryFn: getBoroughs,
  })
