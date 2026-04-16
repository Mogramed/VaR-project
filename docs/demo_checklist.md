# Demo Checklist (Go/No-Go)

Checklist operationnelle pour reduire les risques de demo.

## T-30 min: environnement

- MT5 terminal ouvert sur le compte demo
- API + worker + frontend + MT5 agent demarres
- base migree
- portfolio attendu selectionne

Commandes utiles:

```bash
var-project db upgrade
var-project seed-demo
docker compose up -d --build api worker celery-worker frontend nginx
```

## T-10 min: sante rapide

- `GET /health` repond
- frontend accessible (`/desk`)
- navigation entre pages desk fluide
- dernieres donnees visibles (pas de page vide critique)

Commande de controle:

```bash
var-project demo-smoke --base-url http://127.0.0.1:8000
```

Critere minimum:

- statut global `OK` ou `DEGRADED` acceptable
- pas de blocage total sur les parcours principaux

## T-5 min: parcours a blanc

- `/desk` Overview charge
- `/desk/models` et `/desk/attribution` charges
- `/desk/capital` charge
- `/desk/execution` preview fonctionne
- `/desk/blotter`, `/desk/incidents`, `/desk/reports` accessibles

## T-2 min: securite de demo

- preparer un ticket tres faible en volume
- eviter toute action irreversible non necessaire
- garder un scenario de repli sans submit live

## Go / No-Go

Go si:

- smoke check acceptable
- frontend stable
- flux modeles/capital/preview/reporting fonctionnels

No-Go live submit si:

- MT5 deconnecte de facon persistante
- latence anormale sur execution
- erreurs repetees cote bridge

Dans ce cas:

- faire une demo "risk-first" (overview, modeles, capital, attribution, incidents, reporting)
- expliquer que l'execution live est conditionnee par la disponibilite broker

## Plan B pret a l'emploi (phrase soutenance)

"Le flux live broker est momentanement instable, je bascule sur le scenario controle: on demontre l'analyse VaR, la gouvernance du capital, la reconciliation et le reporting, qui restent pleinement operationnels."

## Anti-risques pendant la soutenance

- ne pas improviser de nouveaux scenarios
- suivre l'ordre du runbook
- limiter les manipulations a forte variabilite
- conserver un rythme constant (10-15 min)

## Post-demo immediat

- noter les points faibles observes
- extraire 3 actions correctives max
- prioriser stabilite et clarte produit avant ajout de features

