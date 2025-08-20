# Research Agent and Multi-Agent Orchestrator
# Continuation of main.py

# Research and Web Intelligence Agent
class ResearchAgent(BaseAgent):
    """Advanced research agent with web scraping and knowledge synthesis"""
    
    def __init__(self, config: AIConfig, memory_manager: AdvancedMemoryManager):
        super().__init__("Research_Agent", config, memory_manager)
        self.capabilities = [
            "web_scraping", "knowledge_synthesis", "fact_checking", 
            "research_planning", "source_validation", "trend_analysis"
        ]
        self.embedding_model = None
        self.knowledge_graph = {}
        
    async def initialize(self):
        """Initialize research models and tools"""
        try:
            logger.info("Initializing Research Agent...")
            
            # Load sentence transformer for semantic similarity
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            
            # Initialize web scraping tools
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            # Knowledge base initialization
            engine = create_engine(self.config.database_url)
            Base.metadata.create_all(engine)
            SessionLocal = sessionmaker(bind=engine)
            self.db_session = SessionLocal()
            
            logger.info("Research Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Research Agent: {e}")
            raise
    
    async def process(self, input_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process research requests with multi-source intelligence"""
        start_time = time.time()
        success = True
        
        try:
            research_type = input_data.get('type', 'general_research')
            query = input_data.get('query', '')
            sources = input_data.get('sources', ['web'])
            depth = input_data.get('depth', 'medium')  # shallow, medium, deep
            
            result = {
                'query': query,
                'research_type': research_type,
                'depth': depth,
                'processed_at': datetime.now().isoformat(),
                'sources_used': [],
                'findings': []
            }
            
            if research_type == 'web_research':
                result['findings'] = await self._conduct_web_research(query, depth)
            elif research_type == 'knowledge_synthesis':
                result['findings'] = await self._synthesize_knowledge(query, input_data)
            elif research_type == 'fact_checking':
                result['findings'] = await self._fact_check(input_data)
            elif research_type == 'trend_analysis':
                result['findings'] = await self._analyze_trends(query, input_data)
            else:
                # General research combining multiple approaches
                result['findings'] = await self._general_research(query, depth)
            
            # Calculate research quality score
            result['quality_score'] = self._calculate_research_quality(result['findings'])
            
            # Store in knowledge base
            await self._store_research_findings(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Research processing failed: {e}")
            success = False
            return {'error': str(e), 'processed_at': datetime.now().isoformat()}
        
        finally:
            response_time = time.time() - start_time
            self.update_metrics(success, response_time)
    
    async def _conduct_web_research(self, query: str, depth: str) -> List[Dict[str, Any]]:
        """Conduct comprehensive web research"""
        findings = []
        
        # Define search strategies based on depth
        search_strategies = {
            'shallow': ['general'],
            'medium': ['general', 'academic', 'news'],
            'deep': ['general', 'academic', 'news', 'specialized', 'social']
        }
        
        strategies = search_strategies.get(depth, ['general'])
        
        for strategy in strategies:
            try:
                urls = await self._get_search_urls(query, strategy)
                for url in urls[:3]:  # Limit per strategy
                    content = await self._scrape_content(url)
                    if content:
                        analysis = await self._analyze_content(content, query)
                        findings.append({
                            'source': url,
                            'strategy': strategy,
                            'content_summary': analysis['summary'],
                            'relevance_score': analysis['relevance'],
                            'credibility_score': analysis['credibility'],
                            'key_points': analysis['key_points'],
                            'timestamp': datetime.now().isoformat()
                        })
            except Exception as e:
                logger.warning(f"Failed to research with strategy {strategy}: {e}")
        
        return findings
    
    async def _get_search_urls(self, query: str, strategy: str) -> List[str]:
        """Get search URLs based on strategy"""
        # This is a simplified implementation
        # In production, you'd integrate with search APIs
        
        base_urls = {
            'general': ['https://example.com/search?q='],
            'academic': ['https://scholar.google.com/scholar?q='],
            'news': ['https://news.google.com/search?q='],
            'specialized': ['https://arxiv.org/search/?query='],
            'social': ['https://twitter.com/search?q=']
        }
        
        # Mock URLs for demonstration
        mock_urls = [
            f"https://example.com/article-{i}-{query.replace(' ', '-')}"
            for i in range(1, 4)
        ]
        
        return mock_urls
    
    async def _scrape_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape and extract content from URL"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract main content
            content = {
                'title': self._extract_title(soup),
                'text': self._extract_main_text(soup),
                'metadata': self._extract_metadata(soup),
                'url': url,
                'scraped_at': datetime.now().isoformat()
            }
            
            return content
            
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title_tag = soup.find('title')
        return title_tag.get_text().strip() if title_tag else "No title found"
    
    def _extract_main_text(self, soup: BeautifulSoup) -> str:
        """Extract main text content"""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Common content selectors
        content_selectors = [
            'article', '.content', '#content', '.post-content',
            '.entry-content', 'main', '.main-content'
        ]
        
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                return content.get_text().strip()
        
        # Fallback to body
        body = soup.find('body')
        return body.get_text().strip() if body else ""
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract page metadata"""
        metadata = {}
        
        # Meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                metadata[name] = content
        
        # Publication date
        date_selectors = [
            'time[datetime]', '.published', '.date', '.post-date'
        ]
        
        for selector in date_selectors:
            date_element = soup.select_one(selector)
            if date_element:
                metadata['publication_date'] = date_element.get('datetime') or date_element.get_text()
                break
        
        return metadata
    
    async def _analyze_content(self, content: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Analyze scraped content for relevance and quality"""
        text = content.get('text', '')
        title = content.get('title', '')
        
        # Calculate relevance using embedding similarity
        query_embedding = self.embedding_model.encode([query])
        content_embedding = self.embedding_model.encode([f"{title} {text[:1000]}"])
        
        relevance_score = float(np.dot(query_embedding, content_embedding.T)[0][0])
        
        # Extract key points using simple extractive method
        sentences = text.split('. ')
        sentence_embeddings = self.embedding_model.encode(sentences)
        
        # Find most similar sentences to query
        similarities = np.dot(query_embedding, sentence_embeddings.T)[0]
        top_indices = np.argsort(similarities)[-3:][::-1]
        key_points = [sentences[i] for i in top_indices if similarities[i] > 0.3]
        
        # Generate summary
        summary = self._generate_extractive_summary(text, max_sentences=3)
        
        # Assess credibility (simplified)
        credibility_score = self._assess_credibility(content)
        
        return {
            'summary': summary,
            'relevance': relevance_score,
            'credibility': credibility_score,
            'key_points': key_points,
            'word_count': len(text.split()),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _generate_extractive_summary(self, text: str, max_sentences: int = 3) -> str:
        """Generate extractive summary"""
        sentences = text.split('. ')
        if len(sentences) <= max_sentences:
            return text
        
        # Simple frequency-based extraction
        word_freq = {}
        words = text.lower().split()
        
        for word in words:
            if word.isalpha() and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Score sentences
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            score = 0
            words_in_sentence = sentence.lower().split()
            for word in words_in_sentence:
                if word in word_freq:
                    score += word_freq[word]
            sentence_scores[i] = score
        
        # Get top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        selected_indices = sorted([idx for idx, _ in top_sentences[:max_sentences]])
        
        return '. '.join([sentences[i] for i in selected_indices]) + '.'
    
    def _assess_credibility(self, content: Dict[str, Any]) -> float:
        """Assess content credibility"""
        score = 0.5  # Base score
        
        metadata = content.get('metadata', {})
        url = content.get('url', '')
        text = content.get('text', '')
        
        # Domain credibility
        if any(domain in url for domain in ['.edu', '.gov', '.org']):
            score += 0.2
        elif any(domain in url for domain in ['.com', '.net']):
            score += 0.1
        
        # Content length (longer = potentially more credible)
        if len(text) > 1000:
            score += 0.1
        
        # Has publication date
        if 'publication_date' in metadata:
            score += 0.1
        
        # Has author information
        if any(key in metadata for key in ['author', 'creator', 'publisher']):
            score += 0.1
        
        return min(score, 1.0)
    
    async def _synthesize_knowledge(self, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Synthesize knowledge from multiple sources"""
        findings = []
        
        # Search existing knowledge base
        existing_knowledge = await self._search_knowledge_base(query)
        
        if existing_knowledge:
            synthesis = await self._create_knowledge_synthesis(existing_knowledge, query)
            findings.append({
                'type': 'knowledge_synthesis',
                'synthesis': synthesis,
                'sources_count': len(existing_knowledge),
                'confidence': synthesis.get('confidence', 0.7),
                'timestamp': datetime.now().isoformat()
            })
        
        # If insufficient knowledge, trigger web research
        if len(existing_knowledge) < 3:
            web_findings = await self._conduct_web_research(query, 'medium')
            findings.extend(web_findings)
        
        return findings
    
    async def _search_knowledge_base(self, query: str) -> List[Dict[str, Any]]:
        """Search existing knowledge base"""
        try:
            # Query database for relevant entries
            knowledge_entries = self.db_session.query(KnowledgeBase).filter(
                KnowledgeBase.topic.contains(query) |
                KnowledgeBase.content.contains(query)
            ).limit(10).all()
            
            results = []
            for entry in knowledge_entries:
                # Calculate semantic similarity if embeddings available
                similarity_score = 0.5  # Default
                if entry.embedding:
                    try:
                        stored_embedding = np.array(json.loads(entry.embedding))
                        query_embedding = self.embedding_model.encode([query])
                        similarity_score = float(np.dot(query_embedding, stored_embedding.reshape(1, -1).T)[0][0])
                    except:
                        pass
                
                results.append({
                    'id': entry.id,
                    'topic': entry.topic,
                    'content': entry.content,
                    'source': entry.source,
                    'confidence': entry.confidence_score,
                    'similarity': similarity_score,
                    'created_at': entry.created_at.isoformat()
                })
            
            # Sort by similarity
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}")
            return []
    
    async def _create_knowledge_synthesis(self, knowledge_entries: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Create a synthesis of knowledge entries"""
        if not knowledge_entries:
            return {'synthesis': 'No relevant knowledge found', 'confidence': 0.0}
        
        # Combine content from top entries
        combined_content = "\n\n".join([
            entry['content'] for entry in knowledge_entries[:5]
        ])
        
        # Generate summary of combined content
        summary = self._generate_extractive_summary(combined_content, max_sentences=5)
        
        # Extract common themes
        themes = self._extract_common_themes(knowledge_entries)
        
        # Calculate confidence based on source count and similarity scores
        avg_similarity = np.mean([entry['similarity'] for entry in knowledge_entries])
        source_count_score = min(len(knowledge_entries) / 10, 1.0)
        confidence = (avg_similarity * 0.7 + source_count_score * 0.3)
        
        return {
            'synthesis': summary,
            'themes': themes,
            'confidence': confidence,
            'source_count': len(knowledge_entries),
            'query': query,
            'created_at': datetime.now().isoformat()
        }
    
    def _extract_common_themes(self, knowledge_entries: List[Dict[str, Any]]) -> List[str]:
        """Extract common themes from knowledge entries"""
        all_text = " ".join([entry['content'] for entry in knowledge_entries])
        
        # Simple keyword extraction
        blob = TextBlob(all_text.lower())
        word_freq = {}
        
        for word, pos in blob.tags:
            if pos in ['NN', 'NNS', 'NNP', 'NNPS'] and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top themes
        themes = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        return [theme[0] for theme in themes if theme[1] > 1]
    
    async def _fact_check(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fact-check claims against reliable sources"""
        claim = params.get('claim', '')
        sources = params.get('preferred_sources', ['academic', 'government'])
        
        findings = []
        
        # Search for supporting/contradicting evidence
        for source_type in sources:
            evidence = await self._search_fact_evidence(claim, source_type)
            if evidence:
                findings.extend(evidence)
        
        # Analyze fact-check results
        fact_check_result = self._analyze_fact_check_results(claim, findings)
        
        return [{
            'type': 'fact_check',
            'claim': claim,
            'verdict': fact_check_result['verdict'],
            'confidence': fact_check_result['confidence'],
            'supporting_evidence': fact_check_result['supporting'],
            'contradicting_evidence': fact_check_result['contradicting'],
            'evidence_count': len(findings),
            'timestamp': datetime.now().isoformat()
        }]
    
    async def _search_fact_evidence(self, claim: str, source_type: str) -> List[Dict[str, Any]]:
        """Search for evidence related to a fact-check claim"""
        # This would integrate with fact-checking APIs and databases
        # For demonstration, we'll simulate evidence gathering
        
        evidence = []
        
        # Mock evidence generation based on source type
        confidence_map = {
            'academic': 0.9,
            'government': 0.85,
            'news': 0.7,
            'social': 0.4
        }
        
        base_confidence = confidence_map.get(source_type, 0.5)
        
        # Simulate evidence
        for i in range(2):
            evidence.append({
                'source_type': source_type,
                'content': f"Evidence {i+1} regarding: {claim}",
                'stance': 'supporting' if i % 2 == 0 else 'contradicting',
                'confidence': base_confidence + (i * 0.05),
                'source_url': f"https://example-{source_type}.com/evidence-{i+1}",
                'found_at': datetime.now().isoformat()
            })
        
        return evidence
    
    def _analyze_fact_check_results(self, claim: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze fact-check evidence to determine verdict"""
        if not evidence:
            return {
                'verdict': 'insufficient_evidence',
                'confidence': 0.0,
                'supporting': [],
                'contradicting': []
            }
        
        supporting = [e for e in evidence if e['stance'] == 'supporting']
        contradicting = [e for e in evidence if e['stance'] == 'contradicting']
        
        # Calculate weighted scores
        support_score = sum([e['confidence'] for e in supporting])
        contradict_score = sum([e['confidence'] for e in contradicting])
        
        total_score = support_score + contradict_score
        
        if total_score == 0:
            verdict = 'insufficient_evidence'
            confidence = 0.0
        elif support_score > contradict_score * 1.5:
            verdict = 'likely_true'
            confidence = support_score / total_score
        elif contradict_score > support_score * 1.5:
            verdict = 'likely_false'
            confidence = contradict_score / total_score
        else:
            verdict = 'disputed'
            confidence = abs(support_score - contradict_score) / total_score
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'supporting': supporting,
            'contradicting': contradicting,
            'support_score': support_score,
            'contradict_score': contradict_score
        }
    
    async def _analyze_trends(self, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze trends related to the query"""
        time_period = params.get('time_period', '1_year')
        trend_type = params.get('trend_type', 'general')
        
        findings = []
        
        # Collect trend data from various sources
        trend_data = await self._collect_trend_data(query, time_period, trend_type)
        
        if trend_data:
            analysis = self._perform_trend_analysis(trend_data, query)
            findings.append({
                'type': 'trend_analysis',
                'query': query,
                'time_period': time_period,
                'trend_direction': analysis['direction'],
                'trend_strength': analysis['strength'],
                'key_patterns': analysis['patterns'],
                'predictions': analysis['predictions'],
                'data_points': len(trend_data),
                'confidence': analysis['confidence'],
                'timestamp': datetime.now().isoformat()
            })
        
        return findings
    
    async def _collect_trend_data(self, query: str, time_period: str, trend_type: str) -> List[Dict[str, Any]]:
        """Collect trend data from various sources"""
        # Mock trend data collection
        # In production, this would integrate with APIs like Google Trends, social media APIs, etc.
        
        trend_data = []
        
        # Generate mock trend data
        import random
        from datetime import timedelta
        
        days_map = {'1_month': 30, '3_months': 90, '1_year': 365, '2_years': 730}
        days = days_map.get(time_period, 365)
        
        base_date = datetime.now() - timedelta(days=days)
        
        for i in range(min(days // 7, 52)):  # Weekly data points
            date = base_date + timedelta(weeks=i)
            
            # Simulate trend values with some randomness and general direction
            base_value = 50 + (i * 0.5) + random.uniform(-10, 10)
            
            trend_data.append({
                'date': date.isoformat(),
                'value': max(0, base_value),
                'query': query,
                'source': trend_type,
                'data_quality': random.uniform(0.7, 1.0)
            })
        
        return trend_data
    
    def _perform_trend_analysis(self, trend_data: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Perform statistical analysis on trend data"""
        if len(trend_data) < 2:
            return {
                'direction': 'insufficient_data',
                'strength': 0.0,
                'patterns': [],
                'predictions': {},
                'confidence': 0.0
            }
        
        values = [item['value'] for item in trend_data]
        
        # Calculate trend direction
        if len(values) > 1:
            slope = (values[-1] - values[0]) / len(values)
            if slope > 1:
                direction = 'increasing'
            elif slope < -1:
                direction = 'decreasing'
            else:
                direction = 'stable'
        else:
            direction = 'stable'
        
        # Calculate trend strength (based on variance and slope)
        variance = np.var(values) if len(values) > 1 else 0
        strength = min(abs(slope) / 10 + variance / 100, 1.0)
        
        # Identify patterns
        patterns = self._identify_patterns(values)
        
        # Simple prediction (linear extrapolation)
        if len(values) >= 3:
            recent_slope = (values[-1] - values[-3]) / 2
            predicted_next = values[-1] + recent_slope
            predictions = {
                'next_period': predicted_next,
                'method': 'linear_extrapolation',
                'confidence': 0.6 if abs(recent_slope) < 5 else 0.4
            }
        else:
            predictions = {'message': 'insufficient_data_for_prediction'}
        
        # Overall confidence
        data_quality = np.mean([item['data_quality'] for item in trend_data])
        confidence = data_quality * min(len(values) / 20, 1.0)
        
        return {
            'direction': direction,
            'strength': strength,
            'patterns': patterns,
            'predictions': predictions,
            'confidence': confidence,
            'slope': slope if 'slope' in locals() else 0,
            'variance': variance
        }
    
    def _identify_patterns(self, values: List[float]) -> List[str]:
        """Identify patterns in trend data"""
        patterns = []
        
        if len(values) < 3:
            return patterns
        
        # Check for seasonality (simplified)
        if len(values) >= 12:
            # Look for repeating patterns
            first_half = values[:len(values)//2]
            second_half = values[len(values)//2:]
            
            if len(first_half) == len(second_half):
                correlation = np.corrcoef(first_half, second_half)[0, 1]
                if correlation > 0.7:
                    patterns.append('seasonal_pattern')
        
        # Check for volatility
        if np.std(values) > np.mean(values) * 0.3:
            patterns.append('high_volatility')
        
        # Check for outliers
        mean_val = np.mean(values)
        std_val = np.std(values)
        outliers = [v for v in values if abs(v - mean_val) > 2 * std_val]
        
        if len(outliers) > len(values) * 0.1:  # More than 10% outliers
            patterns.append('contains_outliers')
        
        # Check for monotonic trends
        if all(values[i] <= values[i+1] for i in range(len(values)-1)):
            patterns.append('monotonic_increasing')
        elif all(values[i] >= values[i+1] for i in range(len(values)-1)):
            patterns.append('monotonic_decreasing')
        
        return patterns
    
    async def _general_research(self, query: str, depth: str) -> List[Dict[str, Any]]:
        """Conduct general research combining multiple approaches"""
        findings = []
        
        # Web research
        web_findings = await self._conduct_web_research(query, depth)
        findings.extend(web_findings)
        
        # Knowledge synthesis
        knowledge_findings = await self._synthesize_knowledge(query, {})
        findings.extend(knowledge_findings)
        
        # If query suggests trend analysis
        trend_keywords = ['trend', 'change', 'growth', 'decline', 'pattern']
        if any(keyword in query.lower() for keyword in trend_keywords):
            trend_findings = await self._analyze_trends(query, {'time_period': '1_year'})
            findings.extend(trend_findings)
        
        return findings
    
    def _calculate_research_quality(self, findings: List[Dict[str, Any]]) -> float:
        """Calculate overall research quality score"""
        if not findings:
            return 0.0
        
        quality_factors = []
        
        for finding in findings:
            # Source diversity
            source_quality = 0.5
            if finding.get('credibility_score'):
                source_quality = finding['credibility_score']
            elif finding.get('confidence'):
                source_quality = finding['confidence']
            
            quality_factors.append(source_quality)
        
        # Calculate weighted average
        base_quality = np.mean(quality_factors)
        
        # Bonus for source diversity
        source_types = set([f.get('strategy', f.get('type', 'unknown')) for f in findings])
        diversity_bonus = min(len(source_types) / 5, 0.2)  # Max 20% bonus
        
        # Bonus for quantity (more findings generally better)
        quantity_bonus = min(len(findings) / 10, 0.1)  # Max 10% bonus
        
        total_quality = min(base_quality + diversity_bonus + quantity_bonus, 1.0)
        
        return total_quality
    
    async def _store_research_findings(self, research_result: Dict[str, Any]):
        """Store research findings in knowledge base"""
        try:
            query = research_result.get('query', '')
            findings = research_result.get('findings', [])
            
            for finding in findings:
                # Create knowledge base entry
                content = json.dumps(finding, default=str)
                
                # Generate embedding for semantic search
                embedding = self.embedding_model.encode([content])
                embedding_json = json.dumps(embedding.tolist())
                
                kb_entry = KnowledgeBase(
                    topic=query,
                    content=content,
                    embedding=embedding_json,
                    source=finding.get('source', 'research_agent'),
                    confidence_score=finding.get('confidence', 0.5)
                )
                
                self.db_session.add(kb_entry)
            
            self.db_session.commit()
            logger.info(f"Stored {len(findings)} research findings for query: {query}")
            
        except Exception as e:
            logger.error(f"Failed to store research findings: {e}")
            self.db_session.rollback()

# Multi-Agent Orchestrator
class MultiAgentOrchestrator:
    """Advanced orchestrator for coordinating multiple AI agents"""
    
    def __init__(self, config: AIConfig):
        self.config = config
        self.memory_manager = AdvancedMemoryManager(config)
        self.agents = {}
        self.task_queue = asyncio.Queue()
        self.result_cache = {}
        self.orchestration_rules = self._load_orchestration_rules()
        self.performance_monitor = PerformanceMonitor()
        
    async def initialize(self):
        """Initialize all agents and orchestration systems"""
        logger.info("Initializing Multi-Agent Orchestrator...")
        
        # Initialize agents
        self.agents['nlu'] = NLUAgent(self.config, self.memory_manager)
        self.agents['generative'] = GenerativeAgent(self.config, self.memory_manager)
        self.agents['research'] = ResearchAgent(self.config, self.memory_manager)
        
        # Initialize all agents
        for agent_name, agent in self.agents.items():
            try:
                await agent.initialize()
                logger.info(f"Agent {agent_name} initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize agent {agent_name}: {e}")
                raise
        
        logger.info("Multi-Agent Orchestrator initialized successfully")
    
    def _load_orchestration_rules(self) -> Dict[str, Any]:
        """Load rules for agent orchestration"""
        return {
            'agent_selection': {
                'question_answering': ['research', 'generative'],
                'text_analysis': ['nlu'],
                'content_generation': ['generative'],
                'research_tasks': ['research', 'nlu'],
                'fact_checking': ['research'],
                'sentiment_analysis': ['nlu'],
                'summarization': ['generative', 'nlu'],
                'translation': ['generative'],
                'creative_writing': ['generative']
            },
            'collaboration_patterns': {
                'sequential': ['nlu', 'generative'],
                'parallel': ['research', 'nlu'],
                'hierarchical': ['research', 'nlu', 'generative']
            },
            'quality_thresholds': {
                'minimum_confidence': 0.6,
                'require_multiple_agents': 0.8,
                'escalation_threshold': 0.9
            }
        }
    
    async def process_request(self, request: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Process user request through appropriate agents"""
        start_time = time.time()
        
        try:
            # Analyze request to determine task type and required agents
            task_analysis = await self._analyze_task(request)
            
            # Select appropriate agents
            selected_agents = self._select_agents(task_analysis)
            
            # Determine orchestration strategy
            strategy = self._determine_strategy(task_analysis, selected_agents)
            
            # Execute based on strategy
            if strategy == 'sequential':
                result = await self._execute_sequential(request, selected_agents, session_id)
            elif strategy == 'parallel':
                result = await self._execute_parallel(request, selected_agents, session_id)
            elif strategy == 'hierarchical':
                result = await self._execute_hierarchical(request, selected_agents, session_id)
            else:
                result = await self._execute_single_agent(request, selected_agents[0], session_id)
            
            # Post-process and validate results
            final_result = await self._post_process_results(result, task_analysis)
            
            # Store conversation in memory
            self.memory_manager.store_conversation(
                session_id, 
                str(request), 
                str(final_result),
                {
                    'task_type': task_analysis['task_type'],
                    'agents_used': selected_agents,
                    'strategy': strategy,
                    'processing_time': time.time() - start_time
                }
            )
            
            return final_result
            
        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'processing_time': time.time() - start_time
            }
    
    async def _analyze_task(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the incoming request to determine task characteristics"""
        text = request.get('text', request.get('query', ''))
        task_type = request.get('task_type', 'auto_detect')
        
        analysis = {
            'original_request': request,
            'text_length': len(text),
            'language': 'en',  # Simplified
            'complexity': 'medium',
            'urgency': 'normal',
            'confidence_required': 0.7
        }
        
        # Auto-detect task type if not specified
        if task_type == 'auto_detect':
            task_type = await self._detect_task_type(text)
        
        analysis['task_type'] = task_type
        
        # Determine complexity based on text length and task type
        if len(text) > 1000 or task_type in ['research_tasks', 'fact_checking']:
            analysis['complexity'] = 'high'
        elif len(text) < 100 and task_type in ['sentiment_analysis', 'text_analysis']:
            analysis['complexity'] = 'low'
        
        # Check for urgency indicators
        urgency_keywords = ['urgent', 'asap', 'quickly', 'immediately', 'fast']
        if any(keyword in text.lower() for keyword in urgency_keywords):
            analysis['urgency'] = 'high'
        
        # Determine required confidence level
        confidence_keywords = ['important', 'critical', 'accurate', 'precise']
        if any(keyword in text.lower() for keyword in confidence_keywords):
            analysis['confidence_required'] = 0.9
        
        return analysis
    
    async def _detect_task_type(self, text: str) -> str:
        """Auto-detect task type from text content"""
        text_lower = text.lower()
        
        # Question patterns
        if any(text.startswith(q) for q in ['what', 'how', 'why', 'when', 'where', 'who']):
            return 'question_answering'
        
        # Research patterns
        research_keywords = ['research', 'find information', 'analyze', 'investigate', 'study']
        if any(keyword in text_lower for keyword in research_keywords):
            return 'research_tasks'
        
        # Generation patterns
        generation_keywords = ['write', 'create', 'generate', 'compose', 'draft']
        if any(keyword in text_lower for keyword in generation_keywords):
            return 'content_generation'
        
        # Analysis patterns
        analysis_keywords = ['analyze', 'sentiment', 'emotion', 'classify', 'categorize']
        if any(keyword in text_lower for keyword in analysis_keywords):
            return 'text_analysis'
        
        # Fact-checking patterns
        fact_keywords = ['fact check', 'verify', 'true', 'false', 'accurate']
        if any(keyword in text_lower for keyword in fact_keywords):
            return 'fact_checking'
        
        # Summarization patterns
        summary_keywords = ['summarize', 'summary', 'brief', 'overview', 'tldr']
        if any(keyword in text_lower for keyword in summary_keywords):
            return 'summarization'
        
        # Creative writing patterns
        creative_keywords = ['story', 'poem', 'creative', 'imagine', 'fiction']
        if any(keyword in text_lower for keyword in creative_keywords):
            return 'creative_writing'
        
        # Default to question answering
        return 'question_answering'
    
    def _select_agents(self, task_analysis: Dict[str, Any]) -> List[str]:
        """Select appropriate agents based on task analysis"""
        task_type = task_analysis['task_type']
        complexity = task_analysis['complexity']
        confidence_required = task_analysis['confidence_required']
        
        # Get base agents for task type
        base_agents = self.orchestration_rules['agent_selection'].get(task_type, ['generative'])
        
        # Add additional agents based on complexity and confidence requirements
        if complexity == 'high' or confidence_required > 0.8:
            if 'research' not in base_agents and task_type != 'creative_writing':
                base_agents.append('research')
            if 'nlu' not in base_agents:
                base_agents.append('nlu')
        
        # Ensure we have valid agents
        available_agents = list(self.agents.keys())
        selected_agents = [agent for agent in base_agents if agent in available_agents]
        
        if not selected_agents:
            selected_agents = ['generative']  # Fallback
        
        return selected_agents
    
    def _determine_strategy(self, task_analysis: Dict[str, Any], selected_agents: List[str]) -> str:
        """Determine orchestration strategy"""
        if len(selected_agents) == 1:
            return 'single'
        
        task_type = task_analysis['task_type']
        complexity = task_analysis['complexity']
        
        # High complexity tasks often benefit from hierarchical processing
        if complexity == 'high' and len(selected_agents) >= 3:
            return 'hierarchical'
        
        # Research tasks often benefit from parallel processing
        if task_type in ['research_tasks', 'fact_checking'] and len(selected_agents) >= 2:
            return 'parallel'
        
        # Most other tasks work well with sequential processing
        return 'sequential'
    
    async def _execute_sequential(self, request: Dict[str, Any], agents: List[str], session_id: str) -> Dict[str, Any]:
        """Execute agents sequentially, passing output from one to the next"""
        current_input = request
        results = {}
        
        for i, agent_name in enumerate(agents):
            agent = self.agents[agent_name]
            
            # Prepare input for current agent
            if i == 0:
                agent_input = self._prepare_agent_input(current_input, agent_name)
            else:
                # Use previous agent's output as context
                agent_input = self._prepare_agent_input(current_input, agent_name, results)
            
            # Execute agent
            agent_result = await agent.process(agent_input)
            results[agent_name] = agent_result
            
            # Prepare input for next agent
            current_input = self._merge_results(current_input, agent_result)
        
        return {
            'strategy': 'sequential',
            'agents_used': agents,
            'results': results,
            'final_output': self._extract_final_output(results, agents),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _execute_parallel(self, request: Dict[str, Any], agents: List[str], session_id: str) -> Dict[str, Any]:
        """Execute agents in parallel and combine results"""
        tasks = []
        
        for agent_name in agents:
            agent = self.agents[agent_name]
            agent_input = self._prepare_agent_input(request, agent_name)
            task = asyncio.create_task(agent.process(agent_input))
            tasks.append((agent_name, task))
        
        # Wait for all agents to complete
        results = {}
        for agent_name, task in tasks:
            try:
                result = await task
                results[agent_name] = result
            except Exception as e:
                logger.error(f"Agent {agent_name} failed: {e}")
                results[agent_name] = {'error': str(e)}
        
        # Combine parallel results
        combined_output = self._combine_parallel_results(results, request)
        
        return {
            'strategy': 'parallel',
            'agents_used': agents,
            'results': results,
            'final_output': combined_output,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _execute_hierarchical(self, request: Dict[str, Any], agents: List[str], session_id: str) -> Dict[str, Any]:
        """Execute agents in hierarchical manner (analysis -> research -> synthesis)"""
        results = {}
        
        # Phase 1: Analysis (NLU agent)
        if 'nlu' in agents:
            nlu_input = self._prepare_agent_input(request, 'nlu')
            results['nlu'] = await self.agents['nlu'].process(nlu_input)
        
        # Phase 2: Research (Research agent with NLU context)
        if 'research' in agents:
            research_input = self._prepare_agent_input(request, 'research', results)
            results['research'] = await self.agents['research'].process(research_input)
        
        # Phase 3: Synthesis (Generative agent with all context)
        if 'generative' in agents:
            generative_input = self._prepare_agent_input(request, 'generative', results)
            results['generative'] = await self.agents['generative'].process(generative_input)
        
        # Create hierarchical output
        final_output = self._create_hierarchical_output(results, request)
        
        return {
            'strategy': 'hierarchical',
            'agents_used': agents,
            'results': results,
            'final_output': final_output,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _execute_single_agent(self, request: Dict[str, Any], agent_name: str, session_id: str) -> Dict[str, Any]:
        """Execute single agent"""
        agent = self.agents[agent_name]
        agent_input = self._prepare_agent_input(request, agent_name)
        result = await agent.process(agent_input)
        
        return {
            'strategy': 'single',
            'agents_used': [agent_name],
            'results': {agent_name: result},
            'final_output': result,
            'timestamp': datetime.now().isoformat()
        }
    
    def _prepare_agent_input(self, request: Dict[str, Any], agent_name: str, 
                           previous_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepare input specific to each agent type"""
        base_input = request.copy()
        
        if agent_name == 'nlu':
            # NLU agent expects text input
            text = request.get('text', request.get('query', ''))
            return text
        
        elif agent_name == 'generative':
            # Generative agent expects structured input
            if previous_results:
                # Add context from previous agents
                context = ""
                if 'nlu' in previous_results:
                    nlu_result = previous_results['nlu']
                    if isinstance(nlu_result, dict):
                        sentiment = nlu_result.get('sentiment', {}).get('dominant', {})
                        entities = nlu_result.get('entities', [])
                        context += f"Sentiment: {sentiment.get('label', 'neutral')}. "
                        if entities:
                            entity_text = ", ".join([e.get('word', '') for e in entities[:3]])
                            context += f"Key entities: {entity_text}. "
                
                if 'research' in previous_results:
                    research_result = previous_results['research']
                    if isinstance(research_result, dict) and 'findings' in research_result:
                        findings_count = len(research_result['findings'])
                        context += f"Research found {findings_count} relevant sources. "
                
                base_input['context'] = context
            
            return base_input
        
        elif agent_name == 'research':
            # Research agent expects structured research request
            research_input = {
                'type': 'general_research',
                'query': request.get('text', request.get('query', '')),
                'depth': 'medium'
            }
            
            # Add context from NLU if available
            if previous_results and 'nlu' in previous_results:
                nlu_result = previous_results['nlu']
                if isinstance(nlu_result, dict):
                    key_phrases = nlu_result.get('key_phrases', [])
                    if key_phrases:
                        research_input['related_terms'] = key_phrases[:5]
            
            return research_input
        
        return base_input
    
    def _merge_results(self, original_input: Dict[str, Any], agent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Merge agent result back into input for next agent"""
        merged = original_input.copy()
        
        # Add agent result as context
        if 'context' not in merged:
            merged['context'] = {}
        
        merged['context']['previous_analysis'] = agent_result
        
        return merged
    
    def _extract_final_output(self, results: Dict[str, Any], agents: List[str]) -> Dict[str, Any]:
        """Extract final output from sequential processing"""
        # Use the last agent's result as primary output
        if agents:
            last_agent = agents[-1]
            if last_agent in results:
                return results[last_agent]
        
        # Fallback to combining all results
        return self._combine_all_results(results)
    
    def _combine_parallel_results(self, results: Dict[str, Any], original_request: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from parallel agent execution"""
        combined = {
            'type': 'parallel_combination',
            'timestamp': datetime.now().isoformat(),
            'original_request': original_request
        }
        
        # Extract key information from each agent
        if 'nlu' in results and isinstance(results['nlu'], dict):
            nlu_result = results['nlu']
            combined['sentiment'] = nlu_result.get('sentiment')
            combined['entities'] = nlu_result.get('entities')
            combined['key_phrases'] = nlu_result.get('key_phrases')
        
        if 'research' in results and isinstance(results['research'], dict):
            research_result = results['research']
            combined['research_findings'] = research_result.get('findings', [])
            combined['research_quality'] = research_result.get('quality_score')
        
        if 'generative' in results and isinstance(results['generative'], dict):
            generative_result = results['generative']
            combined['generated_content'] = generative_result.get('output')
        
        # Create summary combining all insights
        combined['summary'] = self._create_combined_summary(results)
        
        return combined
    
    def _create_hierarchical_output(self, results: Dict[str, Any], original_request: Dict[str, Any]) -> Dict[str, Any]:
        """Create hierarchical output with clear analysis -> research -> synthesis flow"""
        output = {
            'type': 'hierarchical_analysis',
            'timestamp': datetime.now().isoformat(),
            'original_request': original_request
        }
        
        # Phase 1: Analysis Summary
        if 'nlu' in results:
            output['analysis_phase'] = {
                'linguistic_analysis': results['nlu'],
                'phase': 'completed'
            }
        
        # Phase 2: Research Summary
        if 'research' in results:
            output['research_phase'] = {
                'findings': results['research'],
                'phase': 'completed'
            }
        
        # Phase 3: Synthesis
        if 'generative' in results:
            output['synthesis_phase'] = {
                'generated_response': results['generative'],
                'phase': 'completed'
            }
        
        # Create comprehensive final answer
        output['comprehensive_answer'] = self._create_comprehensive_answer(results, original_request)
        
        return output
    
    def _create_combined_summary(self, results: Dict[str, Any]) -> str:
        """Create a summary combining insights from all agents"""
        summary_parts = []
        
        # Add sentiment if available
        if 'nlu' in results:
            nlu_result = results['nlu']
            if isinstance(nlu_result, dict):
                sentiment = nlu_result.get('sentiment', {}).get('dominant', {})
                if sentiment:
                    summary_parts.append(f"Sentiment analysis indicates a {sentiment.get('label', 'neutral')} tone.")
        
        # Add research findings
        if 'research' in results:
            research_result = results['research']
            if isinstance(research_result, dict):
                findings = research_result.get('findings', [])
                if findings:
                    summary_parts.append(f"Research analysis identified {len(findings)} relevant sources.")
        
        # Add generation summary
        if 'generative' in results:
            generative_result = results['generative']
            if isinstance(generative_result, dict):
                output = generative_result.get('output', {})
                if isinstance(output, dict) and 'generated_text' in output:
                    summary_parts.append("Generated comprehensive response based on analysis.")
        
        return " ".join(summary_parts) if summary_parts else "Analysis completed across multiple AI agents."
    
    def _create_comprehensive_answer(self, results: Dict[str, Any], original_request: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive answer integrating all agent results"""
        answer = {
            'question': original_request.get('text', original_request.get('query', '')),
            'comprehensive_response': '',
            'confidence_score': 0.0,
            'sources_consulted': 0,
            'analysis_depth': 'comprehensive'
        }
        
        response_parts = []
        confidence_scores = []
        sources_count = 0
        
        # Integrate research findings
        if 'research' in results:
            research_result = results['research']
            if isinstance(research_result, dict):
                findings = research_result.get('findings', [])
                if findings:
                    sources_count += len(findings)
                    quality_score = research_result.get('quality_score', 0.5)
                    confidence_scores.append(quality_score)
                    
                    # Summarize key research points
                    key_points = []
                    for finding in findings[:3]:  # Top 3 findings
                        if isinstance(finding, dict):
                            summary = finding.get('content_summary', finding.get('summary', ''))
                            if summary:
                                key_points.append(summary)
                    
                    if key_points:
                        research_summary = " ".join(key_points)
                        response_parts.append(f"Based on research findings: {research_summary}")
        
        # Integrate linguistic analysis
        if 'nlu' in results:
            nlu_result = results['nlu']
            if isinstance(nlu_result, dict):
                key_phrases = nlu_result.get('key_phrases', [])
                if key_phrases:
                    response_parts.append(f"Key concepts identified: {', '.join(key_phrases[:5])}")
        
        # Integrate generated content
        if 'generative' in results:
            generative_result = results['generative']
            if isinstance(generative_result, dict):
                output = generative_result.get('output', {})
                if isinstance(output, dict):
                    generated_text = output.get('generated_text', output.get('summary', ''))
                    if generated_text:
                        response_parts.append(generated_text)
                    
                    gen_confidence = output.get('confidence', generative_result.get('confidence', 0.5))
                    confidence_scores.append(gen_confidence)
        
        # Combine all parts
        answer['comprehensive_response'] = " ".join(response_parts)
        answer['sources_consulted'] = sources_count
        answer['confidence_score'] = np.mean(confidence_scores) if confidence_scores else 0.5
        
        return answer
    
    def _combine_all_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback method to combine all results"""
        return {
            'combined_results': results,
            'timestamp': datetime.now().isoformat(),
            'note': 'Combined output from all agents'
        }
    
    async def _post_process_results(self, result: Dict[str, Any], task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process and validate results"""
        # Add metadata
        result['metadata'] = {
            'task_analysis': task_analysis,
            'processing_timestamp': datetime.now().isoformat(),
            'orchestrator_version': '2.0'
        }
        
        # Validate result quality
        quality_score = self._assess_result_quality(result)
        result['quality_assessment'] = {
            'overall_score': quality_score,
            'meets_threshold': quality_score >= self.orchestration_rules['quality_thresholds']['minimum_confidence']
        }
        
        # Add performance metrics
        result['performance_metrics'] = self._collect_performance_metrics()
        
        return result
    
    def _assess_result_quality(self, result: Dict[str, Any]) -> float:
        """Assess overall quality of the orchestrated result"""
        quality_factors = []
        
        # Check if we have actual content
        if 'final_output' in result:
            final_output = result['final_output']
            if isinstance(final_output, dict) and final_output:
                quality_factors.append(0.8)  # Has structured output
            elif isinstance(final_output, str) and len(final_output) > 10:
                quality_factors.append(0.7)  # Has text output
            else:
                quality_factors.append(0.3)  # Minimal output
        
        # Check agent success rates
        if 'results' in result:
            agent_results = result['results']
            successful_agents = sum(1 for r in agent_results.values() 
                                  if isinstance(r, dict) and 'error' not in r)
            total_agents = len(agent_results)
            
            if total_agents > 0:
                success_rate = successful_agents / total_agents
                quality_factors.append(success_rate)
        
        # Check for specific quality indicators
        if 'final_output' in result and isinstance(result['final_output'], dict):
            final_output = result['final_output']
            
            # Check for confidence scores
            if 'confidence_score' in final_output:
                quality_factors.append(final_output['confidence_score'])
            
            # Check for comprehensive content
            if 'comprehensive_response' in final_output:
                response = final_output['comprehensive_response']
                if isinstance(response, str) and len(response) > 50:
                    quality_factors.append(0.8)
        
        return np.mean(quality_factors) if quality_factors else 0.5
    
    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics from all agents"""
        metrics = {
            'agent_performance': {},
            'overall_system_health': 'good'
        }
        
        for agent_name, agent in self.agents.items():
            metrics['agent_performance'][agent_name] = {
                'success_rate': agent.get_success_rate(),
                'average_response_time': agent.performance_metrics['average_response_time'],
                'total_requests': agent.performance_metrics['total_requests']
            }
        
        # Calculate overall system health
        success_rates = [agent.get_success_rate() for agent in self.agents.values()]
        avg_success_rate = np.mean(success_rates) if success_rates else 0
        
        if avg_success_rate > 90:
            metrics['overall_system_health'] = 'excellent'
        elif avg_success_rate > 75:
            metrics['overall_system_health'] = 'good'
        elif avg_success_rate > 50:
            metrics['overall_system_health'] = 'fair'
        else:
            metrics['overall_system_health'] = 'poor'
        
        return metrics
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'orchestrator_status': 'operational',
            'agents': {},
            'memory_manager': {
                'short_term_sessions': len(self.memory_manager.short_term_memory),
                'long_term_entries': len(self.memory_manager.long_term_memory),
                'semantic_concepts': len(self.memory_manager.semantic_memory)
            },
            'performance_summary': self._collect_performance_metrics()
        }
        
        # Check each agent status
        for agent_name, agent in self.agents.items():
            status['agents'][agent_name] = {
                'status': 'operational' if agent.model else 'not_initialized',
                'capabilities': agent.capabilities,
                'performance': {
                    'success_rate': agent.get_success_rate(),
                    'total_requests': agent.performance_metrics['total_requests'],
                    'average_response_time': agent.performance_metrics['average_response_time']
                }
            }
        
        return status

# Performance Monitoring
class PerformanceMonitor:
    """Monitor and track system performance"""
    
    def __init__(self):
        self.metrics = {
            'requests_per_minute': [],
            'response_times': [],
            'error_rates': [],
            'memory_usage': [],
            'timestamp': []
        }
        self.start_time = time.time()
    
    def record_request(self, response_time: float, success: bool):
        """Record a request for performance monitoring"""
        current_time = time.time()
        
        self.metrics['response_times'].append(response_time)
        self.metrics['timestamp'].append(current_time)
        
        # Calculate requests per minute (sliding window)
        window_start = current_time - 60  # Last minute
        recent_requests = [t for t in self.metrics['timestamp'] if t > window_start]
        self.metrics['requests_per_minute'].append(len(recent_requests))
        
        # Keep only recent data (last hour)
        hour_ago = current_time - 3600
        self._cleanup_old_metrics(hour_ago)
    
    def _cleanup_old_metrics(self, cutoff_time: float):
        """Remove metrics older than cutoff time"""
        for key in ['response_times', 'timestamp', 'requests_per_minute']:
            if key in self.metrics:
                # Keep only recent entries
                recent_indices = [i for i, t in enumerate(self.metrics['timestamp']) 
                                if t > cutoff_time]
                if recent_indices:
                    self.metrics[key] = [self.metrics[key][i] for i in recent_indices]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics['response_times']:
            return {'status': 'no_data'}
        
        return {
            'avg_response_time': np.mean(self.metrics['response_times']),
            'median_response_time': np.median(self.metrics['response_times']),
            'p95_response_time': np.percentile(self.metrics['response_times'], 95),
            'current_rpm': self.metrics['requests_per_minute'][-1] if self.metrics['requests_per_minute'] else 0,
            'avg_rpm': np.mean(self.metrics['requests_per_minute']) if self.metrics['requests_per_minute'] else 0,
            'uptime_seconds': time.time() - self.start_time,
            'total_requests': len(self.metrics['response_times'])
        }
