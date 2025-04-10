"use client";

import { useEffect, useState } from "react";
import BoroughCard from "@/components/borough-card";
import {
  getCrimeIndicator,
  getRentIndicator,
  getYouthIndicator,
  toTitleCase,
} from "@/lib/utils";
import { ThumbsDown, ThumbsUp } from "lucide-react";
import Link from "next/link";

type Recommendation = {
  borough: string;
  score: number;
  norm_rent: number;
  norm_crime: number;
  norm_youth: number;
  norm_centrality: number;
};

type BoroughDetail = {
  name: string;
  slug: string;
  image: string;
  norm_rent: number;
  norm_crime: number;
   norm_youth: number;
};

export default function ResultsPage() {
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [boroughs, setBoroughs] = useState<BoroughDetail[]>([]);

  useEffect(() => {
    const stored = sessionStorage.getItem("recommendations");
    if (stored) {
      const parsed: Recommendation[] = JSON.parse(stored);
      const sorted = [...parsed].sort((a, b) => b.score - a.score);
      setRecommendations(sorted);

      Promise.all(
        sorted.map((rec) =>
          fetch(
            `http://127.0.0.1:8000/api/boroughs/${rec.borough
              .toLowerCase()
              .replace(/ /g, "-")}/`
          )
            .then((res) => res.json())
            .catch(() => null)
        )
      ).then((results) => {
        const valid = results.filter((r): r is BoroughDetail => !!r);
        setBoroughs(valid);
      });
    }
  }, []);

  const sendFeedback = async (borough: string, feedback: string) => {
    //  TODO
  };

  const top = recommendations[0];
  const rest = recommendations.slice(1);
  const topBorough = boroughs.find(
    (b) => b.name.toLowerCase() === top?.borough.toLowerCase()
  );

  const rent = getRentIndicator(top?.norm_rent);
  const crime = getCrimeIndicator(top?.norm_crime);
  const youth = getYouthIndicator(top?.norm_youth);

  return (
    <div className="max-w-7xl mx-auto py-10 px-6">
      <h1 className="text-3xl font-bold mb-10">Your Top Borough Matches</h1>

      {top && topBorough && (
        <>
          <Link
            href={`/borough/${topBorough.slug}`}
            className="block mb-4 bg-zinc-900 border border-zinc-700 rounded-3xl p-6 md:flex gap-6 items-center shadow-xl hover:scale-[1.01] transition"
          >
            <div className="relative w-full md:w-1/2 aspect-video overflow-hidden rounded-2xl mb-4 md:mb-0">
              <img
                src={topBorough.image}
                alt={topBorough.name}
                className="w-full h-full object-cover"
              />
            </div>
            <div className="md:w-1/2 text-white">
              <p className="text-sm uppercase tracking-wide text-green-400 font-semibold mb-1">
                Best match!
              </p>
              <h2 className="text-3xl font-bold mb-3">
                {toTitleCase(topBorough.name)}
              </h2>
              <p className="mb-4 text-gray-300">
                This borough matches{" "}
                <span className="text-green-400 font-semibold">
                  {(top.score * 100).toFixed(1)}%
                </span>{" "}
                of your preferences.
              </p>
              <div className="text-sm text-gray-400 flex gap-4 flex-wrap items-center mt-2">
                {rent && (
                  <div className="flex gap-1 items-center">
                    <span className={`flex items-center ${rent.color}`}>
                      {rent.icon}
                    </span>
                    <span>{rent.label}</span>
                  </div>
                )}
                {crime && (
                  <div className="flex gap-1 items-center">
                    <span className={`flex items-center ${crime.color}`}>
                      {crime.icon}
                    </span>
                    <span>{crime.label}</span>
                  </div>
                )}
                {youth && (
                  <div className="flex gap-1 items-center">
                    <span className="flex items-center">{youth.icon}</span>
                    <span>{youth.label}</span>
                  </div>
                )}
              </div>
            </div>
          </Link>

          <div className="flex flex-col mb-14 text-sm text-gray-500 items-end mr-2">
            <div className="flex gap-3 items-center">
              <button
                onClick={() => sendFeedback(topBorough.name, "like")}
                className="flex items-center gap-1 hover:text-green-400 transition text-gray-400"
              >
                <ThumbsUp className="w-4 h-4" strokeWidth={1.5} />
                Yes
              </button>
              <button
                onClick={() => sendFeedback(topBorough.name, "dislike")}
                className="flex items-center gap-1 hover:text-red-400 transition text-gray-400"
              >
                <ThumbsDown className="w-4 h-4" strokeWidth={1.5} />
                No
              </button>
            </div>
          </div>
        </>
      )}

      <h3 className="text-xl font-semibold text-white mb-4">
        Other suggestions
      </h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
        {rest.map((rec) => {
          const details = boroughs.find(
            (b) => b.name.toLowerCase() === rec.borough.toLowerCase()
          );
          if (!details) return null;

          return (
            <BoroughCard
              key={details.slug}
              slug={details.slug}
              name={details.name}
              image={details.image}
              scoreLabel={`Score: ${(rec.score * 100).toFixed(0)}%`}
              norm_rent={details.norm_rent}
              norm_crime={details.norm_crime}
              norm_youth={details.norm_youth}
            />
          );
        })}
      </div>
    </div>
  );
}
