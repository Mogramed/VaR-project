# VaR Risk Desk Platform - Recette Finale

Protocole de test end-to-end sur compte demo MT5.

## 1. Cold start (MT5 disconnected)

- [ ] Demarrer le backend: `uvicorn var_project.api.app:create_app --factory --reload`
- [ ] Demarrer le frontend: `cd frontend && npm run dev`
- [ ] Verifier que `/desk` affiche le dashboard sans erreur
- [ ] Verifier que toutes les pages du nav rail sont accessibles (Overview, MT5 Ops, Universe, Models, Attribution, Capital, Decisions, Dry Run, Stress, Blotter, Reports)
- [ ] Verifier que le badge MT5 dans le header affiche "degraded" ou "disconnected"
- [ ] Verifier que l'API repond: `GET /health` retourne 200

## 2. Connexion MT5

- [ ] Configurer les credentials MT5 demo dans `config/settings.yaml` ou via variables d'environnement
- [ ] Relancer le backend
- [ ] Le badge MT5 passe a "ok" / "connected" dans les 10 secondes
- [ ] La page MT5 Ops affiche les infos du compte (balance, equity, margin)
- [ ] Les holdings live apparaissent dans le dashboard

## 3. Startup import audit

- [ ] Si le compte demo a des positions ouvertes non listees dans `settings.yaml`, verifier qu'un evenement `mt5.startup_import` apparait dans `/audit/recent`
- [ ] Les symboles importes sont listes dans le payload de l'evenement

## 4. Execution preview (Dry Run)

- [ ] Naviguer vers `/desk/execution`
- [ ] Entrer: EURUSD, Buy, 500000 EUR
- [ ] Cliquer "Preview with guard"
- [ ] Verifier que le Broker Ticket affiche: Volume (lots), Price (5 decimales), EUR/lot, Min size
- [ ] Verifier que le VaR Impact affiche: Pre-trade VaR, Post-trade VaR, Delta, Budget utilisation
- [ ] Verifier que le Margin Check affiche: Required, Free after, Equity, Margin level
- [ ] Le guard verdict est APPROVE (sur un compte demo avec marge suffisante)

## 5. Execution submit

- [ ] Cliquer "Send to MT5"
- [ ] Le resultat affiche "EXECUTED" avec les lots remplis
- [ ] Le blotter (`/desk/blotter`) se met a jour dans les 2 cycles de poll (~4s)
- [ ] La position apparait dans le terminal MT5

## 6. Reconciliation

- [ ] Le blotter affiche la position ouverte en statut "match" entre desk et broker
- [ ] Si une position manuelle est ouverte dans le terminal MT5, le blotter detecte un "orphan_live_position"
- [ ] Le bouton "Acknowledge" apparait sur les mismatches
- [ ] Cliquer "Acknowledge" enregistre l'evenement dans l'audit trail

## 7. Stress testing

- [ ] Naviguer vers `/desk/stress`
- [ ] Garder les scenarios par defaut (2008 Crisis, COVID, ECB, Mild)
- [ ] Cliquer "Run stress test"
- [ ] Verifier que les resultats montrent des VaR/ES non-zero et croissants avec le vol_multiplier
- [ ] Ajouter un scenario custom et relancer
- [ ] Verifier que le nouveau scenario apparait dans la table

## 8. Advisory flow (Decisions)

- [ ] Naviguer vers `/desk/decisions`
- [ ] Entrer une proposition de trade large (ex: 5000000 EUR)
- [ ] Verifier que la decision est REDUCE ou REJECT si ca depasse le budget
- [ ] Le suggested_delta_position_eur est affiche
- [ ] Le bloc "Broker preview" indique de passer par Dry Run pour les lots
- [ ] Le bouton "Continue to blotter" mene bien a `/desk/blotter`

## 9. Reports

- [ ] Naviguer vers `/desk/reports`
- [ ] Verifier que le rapport se charge ou peut etre genere
- [ ] Les donnees live (capital, VaR, holdings) sont integrees au rapport

## 10. Route redirect

- [ ] Visiter `/desk/simulation` dans le navigateur
- [ ] Verifier la redirection 301 vers `/desk/blotter`

## 11. Stabilite multi-heures

- [ ] Laisser tourner la plateforme pendant au moins 2 heures
- [ ] Verifier que le live feed reste stable (pas de "stale" persistent)
- [ ] Reconnexion automatique apres une deconnexion MT5 breve
- [ ] Pas de divergence silencieuse entre le compte MT5 et la plateforme
- [ ] Les snapshots et backtests programmés s'executent (verifier `/jobs/status`)

## 12. Cleanup check

- [ ] `grep -r "pnl_mode\|n_update_bars\|PortfolioSpec" src/` retourne zero resultats
- [ ] Le fichier `config/settings.yaml` ne contient plus de section `live:`
- [ ] Aucune reference a `/desk/simulation` dans le frontend (hors page de redirect)
