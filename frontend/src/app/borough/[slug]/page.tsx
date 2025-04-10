'use client'

import type { JSX } from 'react'
import { useEffect, useState } from 'react'
import { useParams } from 'next/navigation'
import Image from 'next/image'
import {
  getCrimeIndicator,
  getRentIndicator,
  getYouthIndicator,
  toTitleCase,
} from '@/lib/utils'

type Borough = {
  name: string
  slug: string
  image: string
  norm_rent: number
  norm_crime: number
  norm_youth: number
}

type Indicator = {
  icon: JSX.Element
  color?: string
  label: string
}

type BoroughData = {
  rent: Indicator | null
  crime: Indicator | null
  youth: Indicator | null
}

export default function BoroughDetailPage() {
  const { slug } = useParams()
  const [borough, setBorough] = useState<Borough | null>(null)
  const [boroughData, setBoroughData] = useState<BoroughData>()

  useEffect(() => {
    fetch(`http://127.0.0.1:8000/api/boroughs/${slug}/`)
      .then((res) => res.json())
      .then((data) => {
        setBorough(data)
        const rent = getRentIndicator(data.norm_rent)
        const crime = getCrimeIndicator(data.norm_crime)
        const youth = getYouthIndicator(data.norm_youth)
        setBoroughData({ rent, crime, youth })
        console.log('Borough data:', data, { rent, crime, youth })
      })
      .catch((err) => console.error('Error fetching borough:', err))
  }, [slug])

  if (!borough) return <div className="p-10 text-white">Loading...</div>

  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <div className="relative">
        <div className="relative z-10 h-[300px] overflow-hidden rounded-[2rem]">
          <Image
            src={borough.image ?? '/images/london-background.png'}
            alt={borough.name}
            fill
            className="object-cover"
          />
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-black bg-opacity-40 p-8">
            <h1 className="text-5xl font-bold text-white">
              {toTitleCase(borough.name)}
            </h1>
          </div>
        </div>

        {/* Cards */}
        <div className="relative z-50 -mt-10 flex justify-center gap-6">
          {boroughData?.rent && (
            <div className="rounded-xl border border-gray-800 bg-[#111111] px-6 py-4 text-center text-white shadow-md">
              <div className="flex gap-2">
                <span className={`flex items-center ${boroughData.rent.color}`}>
                  {boroughData.rent.icon}
                </span>
                <span> {boroughData.rent.label}</span>
              </div>
            </div>
          )}
          {boroughData?.crime && (
            <div className="flex flex-col items-center justify-center rounded-xl border border-gray-800 bg-[#111111] px-6 py-4 text-center text-white shadow-md">
              <div className="flex gap-2">
                <span
                  className={`flex items-center ${boroughData.crime.color}`}
                >
                  {boroughData.crime.icon}
                </span>
                <span> {boroughData.crime.label}</span>
              </div>
            </div>
          )}
          {boroughData?.youth && (
            <div className="flex flex-col items-center justify-center rounded-xl border border-gray-800 bg-[#111111] px-6 py-4 text-center text-white shadow-md">
              <div className="flex gap-2">
                <span className="flex items-center">
                  {boroughData.youth.icon}
                </span>
                <span> {boroughData.youth.label}</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
