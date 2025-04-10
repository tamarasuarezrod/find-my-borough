import { api } from '@/lib/axios'
import { Borough } from '@/types/borough'
import { useQuery } from '@tanstack/react-query'

export const getBoroughs = async (): Promise<Borough[]> => {
  const res = await api.get('/boroughs/')
  return res.data
}

export const useBoroughs = () =>
  useQuery({
    queryKey: ['boroughs'],
    queryFn: getBoroughs,
  })
