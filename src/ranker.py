import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict, deque
from typing import List, Dict, Tuple
import re

class IntelligentConversationTracker:
    def __init__(self, model_path: str, device_map: str = "auto", max_memory: Dict = {0: "3500MB"}):
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=device_map,
            max_memory=max_memory
        )
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Estado conversacional
        self.history = deque(maxlen=20)
        self.active_topic = None
        self.topics = defaultdict(lambda: {
            'tokens': set(),
            'pheromones': defaultdict(float),
            'messages': [],
            'last_used': 0
        })
        
        # Parámetros optimizados
        self.topic_threshold = 0.45
        self.pheromone_decay = 0.85
        self.temporal_decay = 0.92
        self.base_pheromone = 1.2
        self.max_context_length = 4096
        
        # Stopwords mejoradas
        self.stop_tokens = {
            '?', ',', '.', 'Ġde', 'Ġel', 'y', 'es', 'son', 'en', 'a', 
            'se', 'que', 'por', 'Ġla', 'Ġlos', 'Ġdel', 'Ġun', 'Ġuna'
        }

        self.topic_cooldown = 2  # Pasos mínimos antes de cambiar de tópico
        self.last_topic_change = 0
        self.current_topic_strength = 1.5

    def _preprocess(self, text: str) -> List[str]:
        """Tokenización avanzada con limpieza semántica"""
        text = re.sub(r'[^\w\sáéíóúñ]', '', text.lower())
        tokens = self.tokenizer.tokenize(text)
        return [t for t in tokens if t not in self.stop_tokens and len(t) > 2]

    def _topic_similarity(self, tokens: List[str], topic: str) -> float:
        """Calcula similitud con un tópico existente"""
        topic_data = self.topics[topic]
        if not topic_data['tokens']:
            return 0.0
        common = len(set(tokens) & topic_data['tokens'])
        return common / (len(tokens) + len(topic_data['tokens']))

    def _detect_topic(self, tokens: List[str]) -> Tuple[str, bool]:
        """Detección de tópico con histéresis reforzada"""
        if len(self.history) - self.last_topic_change < self.topic_cooldown:
            return self.active_topic, False

        best_topic = self.active_topic
        best_score = 0.0
        new_topic = False
        
        # Cálculo de similitud mejorado
        for topic in self.topics:
            topic_tokens = self.topics[topic]['tokens']
            common = len(set(tokens) & topic_tokens)
            total = len(tokens) + len(topic_tokens)
            score = (2 * common) / total if total > 0 else 0.0
            score *= (self.topics[topic]['last_used'] / len(self.history))
            
            if score > best_score:
                best_score = score
                best_topic = topic

        # Umbral dinámico basado en actividad reciente
        dynamic_threshold = self.topic_threshold * self.current_topic_strength
        if best_score < dynamic_threshold or best_topic != self.active_topic:
            best_topic = f"topic_{len(self.topics)}"
            new_topic = True
            self.last_topic_change = len(self.history)
            self.current_topic_strength = 1.0
        else:
            self.current_topic_strength *= 1.1

        return best_topic, new_topic

    def get_context(self, query: str, top_k: int = 3, threshold: float = 0.35) -> List[str]:
        """Devuelve contexto que supere el umbral de relevancia"""
        query_tokens = self._preprocess(query)
        current_topic, _ = self._detect_topic(query_tokens)
        
        # Obtener candidatos del tópico actual
        candidates = [
            msg for msg in reversed(self.history) 
            if msg['topic'] == current_topic
        ][:top_k*2]
        
        # Obtener mensajes clasificados con sus scores
        ranked_messages = self._rank_messages_with_scores(query, query_tokens, candidates)

        if len(ranked_messages)!=0:
            threshold=0
            i=0
            for score, msg in ranked_messages:
                print(score)
                threshold+=score
                i+=1
            threshold=threshold/i
        
        # Filtrar por umbral y límite de tokens
        context = []
        current_length = 0
        for score, msg in ranked_messages:
            if score < threshold and len(context) > 0:
                break  # Solo permitir un mensaje por debajo del umbral al inicio
                
            msg_tokens = self.tokenizer.tokenize(msg['text'])
            if current_length + len(msg_tokens) <= self.max_context_length:
                context.append(msg['text'])
                current_length += len(msg_tokens)
                
            if len(context) >= top_k:
                break
        
        return context

    def add_message(self, message: str):
        """Procesa un nuevo mensaje actualizando el estado conversacional"""
        tokens = self._preprocess(message)
        current_topic, is_new_topic = self._detect_topic(tokens)
        
        # Manejo de cambio de tópico
        if is_new_topic:
            self.active_topic = current_topic
            self._decay_other_topics()
        
        # Actualizar feromonas del tópico actual
        self._update_pheromones(tokens, boost_factor=2.0 if is_new_topic else 1.0)
        
        # Registrar mensaje
        self.history.append({
            'text': message,
            'tokens': tokens,
            'topic': current_topic,
            'position': len(self.history)
        })
        self.topics[current_topic]['messages'].append(len(self.history)-1)
        self.topics[current_topic]['tokens'].update(tokens)
        self.topics[current_topic]['last_used'] = len(self.history)
    
    def _rank_messages_with_scores(self, query: str, query_tokens: List[str], candidates: List[Dict]) -> List[Tuple[float, Dict]]:
        """Clasificación con retorno de scores completos"""
        if not candidates:
            return []
        
        pairs = [[query, msg['text']] for msg in candidates]
        with torch.no_grad():
            scores = self.model.compute_score(pairs, max_length=1024, doc_type="text")
        
        # Manejo de tipos de scores
        if isinstance(scores, float):
            scores = [scores]
        elif torch.is_tensor(scores):
            scores = scores.cpu().numpy().tolist()
        
        ranked = []
        for idx, msg in enumerate(candidates):
            semantic = scores[idx] if idx < len(scores) else 0.0
            temporal = 0.9 ** (len(self.history) - msg['position'])
            pheromone = sum(self.topics[msg['topic']]['pheromones'].get(t, 0) for t in query_tokens)
            relevance = (0.2 * semantic) + (0.6 * pheromone) + (0.2 * temporal)
            ranked.append((relevance, msg))
        
        # Ordenar descendente
        #print(sorted(ranked, key=lambda x: x[0], reverse=True))
        return sorted(ranked, key=lambda x: x[0], reverse=True)

    def _rank_messages(self, query: str, query_tokens: List[str], candidates: List[Dict], k: int) -> List[str]:
        """Clasificación jerárquica de mensajes candidatos"""
        if not candidates:
            return []
        
        # Preselección semántica
        pairs = [[query, msg['text']] for msg in candidates]
        with torch.no_grad():
            scores = self.model.compute_score(pairs, max_length=1024, doc_type="text")
        
        # Manejo robusto de tipos de retorno
        if isinstance(scores, float):
            scores = [scores]
        elif torch.is_tensor(scores):
            scores = scores.cpu().numpy().tolist()
        
        # Asegurar que scores sea una lista
        if not isinstance(scores, list):
            scores = [scores]
        
        # Ponderación multi-factor
        ranked = []
        for idx, msg in enumerate(candidates):
            semantic = scores[idx] if idx < len(scores) else 0.0
            temporal = 0.7 ** (len(self.history) - msg['position'])
            pheromone = sum(self.topics[msg['topic']]['pheromones'].get(t, 0) for t in query_tokens)
            relevance = (0.3 * semantic) + (0.5 * pheromone) + (0.2 * temporal)
            ranked.append((relevance, msg))
        
        # Selección y construcción de contexto
        ranked.sort(reverse=True, key=lambda x: x[0])
        context = []
        current_length = 0
        for score, msg in ranked[:k]:
            tokens = self.tokenizer.tokenize(msg['text'])
            if current_length + len(tokens) <= self.max_context_length:
                context.append(msg['text'])
                current_length += len(tokens)
        return context

    def _update_pheromones(self, tokens: List[str], boost_factor: float = 1.0):
        """Actualiza feromonas con decaimiento adaptativo"""
        topic_data = self.topics[self.active_topic]
        
        # Decaimiento de feromonas existentes
        for token in topic_data['pheromones']:
            topic_data['pheromones'][token] *= self.pheromone_decay
        
        # Refuerzo de tokens actuales
        for token in tokens:
            topic_data['pheromones'][token] = min(
                topic_data['pheromones'][token] + (self.base_pheromone * boost_factor),
                5.0  # Límite máximo
            )

    def _decay_other_topics(self):
        """Aplica decaimiento agresivo a tópicos inactivos"""
        current_step = len(self.history)
        for topic in self.topics:
            if topic != self.active_topic:
                steps_inactive = current_step - self.topics[topic]['last_used']
                decay = self.pheromone_decay ** steps_inactive
                for token in self.topics[topic]['pheromones']:
                    self.topics[topic]['pheromones'][token] *= decay

    def get_active_topic(self) -> str:
        return self.active_topic
    

tracker = IntelligentConversationTracker("../models/jina-reranker-m0")

texts = [
    "Que reportes se envian al IPAB?",
    "Cuales de ellos son por evento?",
    "Cuantos son en total",
    "Cuantas validaciones estan activas actualmente?",
    "Cuales de ellas son de tipo INTRA?",
    "Me recuerdas a cuales son los reportes del IPAB?",
    "Que reportes se son de BM?",
    "Algunos de ellos son trimestrales?"
]

for text in texts:
    print(f"\nUsuario: {text}")
    context = tracker.get_context(text)
    print(f"Contexto relevante: {context}")
    tracker.add_message(text)