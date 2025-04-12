'use client'

import { useParams } from 'next/navigation'
import Image from 'next/image'
import {
  getCentralityIndicator,
  getCrimeIndicator,
  getRentIndicator,
  getYouthIndicator,
  toTitleCase,
} from '@/lib/utils'
import { useBoroughBySlug } from '@/services/get-borough-detail'
import CommunityRatings from './community-ratings'

export default function BoroughDetailPage() {
  const { slug } = useParams()
  const { data: borough, isLoading, isError } = useBoroughBySlug(slug as string)

  const rent = getRentIndicator(borough?.norm_rent)
  const crime = getCrimeIndicator(borough?.norm_crime)
  const youth = getYouthIndicator(borough?.norm_youth)
  const centrality = getCentralityIndicator(borough?.norm_centrality)

  if (isLoading) {
    return <div className="text-center text-gray-400">Loading borough...</div>
  }

  if (isError || !borough) {
    return (
      <div className="text-center text-red-400">Failed to load the borough</div>
    )
  }

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
            <h1 className="break-words text-center text-2xl font-bold text-white sm:text-4xl md:text-5xl">
              {toTitleCase(borough.name)}
            </h1>
          </div>
        </div>

        {/* Indicators */}
        <div className="relative z-50 -mt-10 grid grid-cols-2 gap-4 px-4 sm:flex sm:justify-center sm:gap-6">
          {centrality && (
            <div className="flex items-center justify-center gap-2 rounded-xl border border-gray-800 bg-[#111111] px-4 py-3 text-white shadow-md">
              <span className="flex items-center">{centrality.icon}</span>
              <span>{centrality.label}</span>
            </div>
          )}
          {rent && (
            <div className="flex items-center justify-center gap-2 rounded-xl border border-gray-800 bg-[#111111] px-4 py-3 text-white shadow-md">
              <span className={`flex items-center ${rent.color}`}>
                {rent.icon}
              </span>
              <span>{rent.label}</span>
            </div>
          )}
          {crime && (
            <div className="flex items-center justify-center gap-2 rounded-xl border border-gray-800 bg-[#111111] px-4 py-3 text-white shadow-md">
              <span className={`flex items-center ${crime.color}`}>
                {crime.icon}
              </span>
              <span>{crime.label}</span>
            </div>
          )}
          {youth && (
            <div className="flex items-center justify-center gap-2 rounded-xl border border-gray-800 bg-[#111111] px-4 py-3 text-white shadow-md">
              <span className="flex items-center">{youth.icon}</span>
              <span>{youth.label}</span>
            </div>
          )}
        </div>

        <CommunityRatings boroughSlug={slug as string} />
      </div>
    </div>
  )
}
