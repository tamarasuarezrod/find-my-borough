import { api } from '@/lib/axios'
import { useQuery } from '@tanstack/react-query'

export type MatchOption = {
  label: string
  value: string | number | boolean
}

export type MatchQuestion = {
  id: string
  title: string
  description: string
  question_type: string
  options: MatchOption[]
}

export const getMatchQuestions = async (): Promise<MatchQuestion[]> => {
  const res = await api.get('/match/questions/')
  return res.data
}

export const useMatchQuestions = () =>
  useQuery({
    queryKey: ['match-questions'],
    queryFn: getMatchQuestions,
  })
