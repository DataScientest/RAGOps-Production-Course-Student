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

Pour commencer le chapitre N avec la correction du chapitre N−1 :

```bash
git clone -b chapter-2 https://github.com/DataScientest/RAGOps-Production-Course-Student.git ragops
cd ragops
uv sync
```

Il faut Docker avec Compose v2, uv, et les deux valeurs de la gateway du cours. Recréez ensuite `.env`, et `users.json` à partir du chapitre 3, comme le décrit l'encadré « Pour reprendre ici » du chapitre.

Les fichiers `.env` et `users.json` sont des secrets : ils ne sont jamais suivis par Git.
