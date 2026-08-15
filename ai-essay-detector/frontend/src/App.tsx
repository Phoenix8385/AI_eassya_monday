import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Landing } from './components/Landing'
import { Analyzer } from './components/Analyzer'
import { useCanAnimate } from './lib/useCanAnimate'
import './App.css'

type View = 'landing' | 'analyzer'

export default function App() {
  const [view, setView] = useState<View>('landing')
  const first = useRef(true)
  const headingRef = useRef<HTMLDivElement>(null)
  const canAnimate = useCanAnimate()

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
      {/*
        `mode="wait"` so the outgoing view finishes leaving before the next one
        arrives — crossfading two full-page layouts on top of each other reads
        as a glitch rather than a transition.

        `initial={false}` means the very first render is not treated as an
        entrance: the landing page must be on screen immediately, not faded in
        by the router shim. And when entrances cannot run at all, the views are
        rendered bare, with no motion wrapper holding an opacity the animation
        would have had to remove.
      */}
      {canAnimate ? (
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={view}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.26, ease: [0.22, 1, 0.36, 1] }}
          >
            {view === 'landing' ? (
              <Landing onStart={() => setView('analyzer')} />
            ) : (
              <Analyzer onBack={() => setView('landing')} />
            )}
          </motion.div>
        </AnimatePresence>
      ) : view === 'landing' ? (
        <Landing onStart={() => setView('analyzer')} />
      ) : (
        <Analyzer onBack={() => setView('landing')} />
      )}
    </div>
  )
}
