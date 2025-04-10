'use client'

import { Check } from 'lucide-react'
import { Toaster } from 'react-hot-toast'

export default function ToastProvider() {
  return (
    <Toaster
      position="top-right"
      toastOptions={{
        duration: 2000,
        style: {
          background: '#18181b',
          color: '#fff',
          borderRadius: '12px',
          padding: '14px 20px',
          fontSize: '14px',
          boxShadow: '0 4px 16px rgba(0, 0, 0, 0.5)',
        },
        success: {
          icon: <Check className="h-5 w-5 text-white" />,
          iconTheme: {
            primary: '#10b981',
            secondary: '#18181b',
          },
          style: {
            background: '#10b981',
            color: '#fff',
            boxShadow: '0 4px 16px rgba(16, 185, 129, 0.4)',
          },
        },
        error: {
          iconTheme: {
            primary: '#ef4444',
            secondary: '#18181b',
          },
          style: {
            background: '#ef4444',
            color: '#fff',
            boxShadow: '0 4px 16px rgba(239, 68, 68, 0.4)',
          },
        },
      }}
    />
  )
}
