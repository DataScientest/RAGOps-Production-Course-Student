# RAGOps — mettre en production : dépôt de pratique

Ce dépôt accompagne le module « RAGOps — mettre en production ». Le module reprend l'assistant sur les congés construit dans « RAG — méthode et techniques », et le fait tourner en service.

## Les branches

- `main` : le point de départ, c'est-à-dire le projet à la fin du module « RAG — méthode et techniques ».
- `chapter-N` : le projet à la fin du chapitre N.

| Branche | Fin du chapitre |
|---|---|
| `chapter-1` | Du script au service |
| `chapter-2` | La vie de l'index |
| `chapter-3` | Sécurité |
| `chapter-4` | Un seuil de qualité |
| `chapter-5` | Surveiller en production |

## Reprendre à un chapitre

Il faut Docker avec Compose v2, uv, et les deux valeurs de la gateway du cours.

Pour commencer le chapitre N, partez de la branche `chapter-(N−1)`, ou de `main` pour le chapitre 1. Exemple pour le chapitre 3 :

```bash
git clone -b chapter-2 https://github.com/DataScientest/RAGOps-Production-Course-Student.git ragops
cd ragops
uv sync
cp .env.example .env
```

Dans `.env`, remplacez chaque `remplacez-moi` : les deux valeurs de la gateway, puis, à partir de la branche `chapter-1`, des valeurs tirées au hasard. Le fichier indique les exercices qui expliquent chaque ligne.

À partir de la branche `chapter-3`, créez aussi les jetons des utilisateurs (exercice 3.3) : copiez `users.example.json` en `users.json` et remplacez chaque `remplacez-moi` par une valeur au hasard.

Les fiches de Service-Public ne sont pas dans Git. Téléchargez-les (exercice 1.4 du module 1) :

```bash
mkdir -p data/raw
curl -L -o data/raw/vosdroits-latest.zip https://lecomarquage.service-public.gouv.fr/vdd/3.5/part/zip/vosdroits-latest.zip
unzip -q -o data/raw/vosdroits-latest.zip -d data/raw/xml
```

L'encadré « Pour reprendre ici » du chapitre donne ensuite les commandes qui relancent les services et reconstruisent l'index.

Pour tout arrêter et libérer les ports : `docker compose down -v`.

Les fichiers `.env` et `users.json` sont des secrets : ils ne sont jamais suivis par Git.
