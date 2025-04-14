'use client'

import { getBoroughs } from '@/services/get-boroughs'
import { Borough } from '@/types/borough'
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
}

const BoroughsContext = createContext<BoroughsContextType>({
  boroughs: null,
  loading: false,
})

export const useBoroughsContext = () => useContext(BoroughsContext)

export function BoroughsProvider({ children }: { children: ReactNode }) {
  const [boroughs, setBoroughs] = useState<Borough[] | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await getBoroughs()
        setBoroughs(data)
        setLoading(false)
      } catch {}
    }

    fetchData()
  }, [])

  return (
    <BoroughsContext.Provider value={{ boroughs, loading }}>
      {children}
    </BoroughsContext.Provider>
  )
}
