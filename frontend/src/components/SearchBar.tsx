// frontend/src/components/SearchBar.tsx
// HP5 — Address hash lookup → triggers WalletDetail

import { useRef, useState } from 'react'

interface Props {
  onSearch: (address: string) => void
}

export function SearchBar({ onSearch }: Props) {
  const [value, setValue] = useState('')
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const trimmed = value.trim()
    if (!trimmed) { setError('Enter an address hash'); return }
    if (trimmed.length < 8) { setError('Hash too short — enter at least 8 characters'); return }
    setError(null)
    onSearch(trimmed)
  }

  return (
    <form onSubmit={handleSubmit}>
      <div className="search-bar">
        <input
          ref={inputRef}
          id="search-address"
          autoComplete="off"
          spellCheck={false}
          placeholder="Search address hash…"
          value={value}
          onChange={e => { setValue(e.target.value); setError(null) }}
        />
        <button type="submit">Go</button>
      </div>
      {error && (
        <div style={{ fontSize: '0.72rem', color: 'var(--risk-high)', marginTop: 4, paddingLeft: 2 }}>
          {error}
        </div>
      )}
    </form>
  )
}
