'use client'

import { Borough, getBoroughs } from '@/services/get-boroughs'
import {
  createContext,
  useContext,
  useEffect,
  useState,
  ReactNode,
} from 'react'

type BoroughsContextType = {
  boroughs: Borough[] | null
  loading: boolean
  error: string | null
}

const BoroughsContext = createContext<BoroughsContextType>({
  boroughs: null,
  loading: false,
  error: null,
})

export const useBoroughsContext = () => useContext(BoroughsContext)

export function BoroughsProvider({ children }: { children: ReactNode }) {
  const [boroughs, setBoroughs] = useState<Borough[] | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await getBoroughs()
        setBoroughs(data)
      } catch (err: any) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  return (
    <BoroughsContext.Provider value={{ boroughs, loading, error }}>
      {children}
    </BoroughsContext.Provider>
  )
}
