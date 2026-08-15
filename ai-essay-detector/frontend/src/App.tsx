import { useEffect, useRef, useState } from 'react'
import { Landing } from './components/Landing'
import { Analyzer } from './components/Analyzer'
import './App.css'

type View = 'landing' | 'analyzer'

export default function App() {
  const [view, setView] = useState<View>('landing')
  const first = useRef(true)
  const headingRef = useRef<HTMLDivElement>(null)

  // Two views with no router means no navigation event, so focus would stay
  // wherever the last click left it. Moving it to the top of the new view is
  // what a real page change would have done.
  useEffect(() => {
    if (first.current) {
      first.current = false
      return
    }
    headingRef.current?.focus()
    window.scrollTo({ top: 0, behavior: 'auto' })
  }, [view])

  return (
    <div className="app" ref={headingRef} tabIndex={-1}>
      {view === 'landing' ? (
        <Landing onStart={() => setView('analyzer')} />
      ) : (
        <Analyzer onBack={() => setView('landing')} />
      )}
    </div>
  )
}
