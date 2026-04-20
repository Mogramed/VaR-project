# Demo Runbook (Soutenance)

Runbook de presentation oriente produit, pour une demonstration propre et professionnelle de la plateforme VaR.

## Objectif de la demo

Montrer en 10 a 15 minutes que la plateforme est:

- stable en conditions live/semi-live
- coherente financierement autour de la VaR
- capable d'aller du risque a l'action (decision, execution, controle, reporting)

## Message principal a repeter

"La plateforme est la couche de pilotage risque au-dessus de MT5: elle mesure, decide, execute, reconcilie et documente."

## Preparation avant entree en salle

- Stack demarree (API, worker, frontend, MT5 agent, terminal MT5)
- Portfolio actif configure en mode `live_mt5`
- En mode live, aucun fallback portefeuille config n est applique: sans donnees MT5 exploitables, le calcul live est bloque (fail-fast).
- Etat de base propre: pas d'incident bloquant non traite
- Commande de verification rapide executee:

```bash
var-project demo-smoke --base-url http://127.0.0.1:8000
```

## Deroule recommande (10-15 min)

## 1) Introduction et posture globale (2 min)

Ecran: `/desk` (Overview)

Ce que tu montres:

- posture risque en direct
- capital et exposition consolides
- coherence des signaux live

Ce que tu dis:

- "On suit le desk en continu: risque, capital, exposition."
- "La VaR guide la decision, pas seulement le PnL."

## 2) Coeur finance: modeles VaR et validation (2-3 min)

Ecran: `/desk/models`, puis `/desk/attribution`

Ce que tu montres:

- champion/challenger
- qualite de validation (exception rate, ES tail ratio)
- contribution au risque par instrument

Ce que tu dis:

- "On ne prend pas un modele par defaut: on le challenge."
- "On explique le risque par contributions, pas en boite noire."

## 3) Discipline capital et limites (1-2 min)

Ecran: `/desk/capital`

Ce que tu montres:

- budget de risque consomme/restant
- headroom
- alertes de depassement si presentes

Ce que tu dis:

- "La VaR est reliee au budget de capital."
- "On controle la prise de risque avant l'execution."

## 4) Flux decision -> execution -> controle (3-4 min)

Ecran: `/desk/execution`, puis `/desk/blotter`

Ce que tu fais:

1. Preview d'un petit trade (ex: `EURUSD` exposition faible)
2. Montre impact marge + VaR post-trade
3. Submit (si environnement demo pret)
4. Va sur blotter pour verifier statut et remplissage

Ce que tu dis:

- "La decision est pre-trade et quantitative."
- "Apres execution, on verifie immediatement la coherence broker/desk."

## 5) Gestion d'incident et tracabilite (1-2 min)

Ecran: `/desk/incidents`

Ce que tu montres:

- incident type drift/manual/orphan
- changement de statut (investigating/resolved)
- audit trail

Ce que tu dis:

- "On transforme un mismatch technique en workflow operable."
- "Chaque action est tracee."

## 6) Reporting final (1-2 min)

Ecran: `/desk/reports`

Ce que tu montres:

- generation snapshot/backtest/report
- coherence vocabulaire UI/PDF (exposure, holdings, lots)

Ce que tu dis:

- "Le reporting reprend l'etat reel du desk, pas une photo statique."

## Plan B pendant la soutenance

Si MT5 live devient instable:

- rester en mode frontend produit
- privilegier preview, modeles, capital, attribution, reporting
- expliquer que l'architecture est tolerante via snapshots et historiques recents

Formulation pro:

"Le live broker est momentanement degrade, la plateforme garde une continuite operationnelle grace aux snapshots et a la couche de controle risque."

Note: en mode live strict, les nouveaux calculs live restent bloques tant que le broker MT5 ne fournit pas de book exploitable.

## Cloture (30 sec)

Conclusion recommandee:

- "On a une plateforme stable de pilotage VaR, connectee au flux d'execution."
- "La valeur ajoutee est l'integration finance + operationnel + tracabilite."
