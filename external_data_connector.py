"""
connectors/external_data_connector.py
کانکتور داده‌های خارجی برای تکمیل تحلیل‌های بازار
"""

import aiohttp
import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import threading
from dataclasses import dataclass
import pandas as pd

@dataclass
class DataSource:
    """کلاس تعریف منابع داده"""
    name: str
    url: str
    api_key: Optional[str] = None
    rate_limit: int = 60  # درخواست در دقیقه
    timeout: int = 10
    headers: Optional[Dict] = None
    active: bool = True

class ExternalDataConnector:
    """
    کانکتور برای دریافت داده‌های خارجی از منابع مختلف:
    - CoinGecko (قیمت‌ها، market cap، fear & greed)
    - CoinMarketCap (آمار بازار)
    - Alternative.me (Fear & Greed Index)
    - Glassnode (On-chain metrics)
    - News APIs (اخبار کریپتو)
    - Social sentiment (Twitter, Reddit)
    """

    def __init__(self):
        self.lock = threading.Lock()
        
        # منابع داده
        self.data_sources = {
            'coingecko': DataSource(
                name='CoinGecko',
                url='https://api.coingecko.com/api/v3',
                rate_limit=50,  # 50 calls per minute for free tier
                headers={'accept': 'application/json'}
            ),
            'feargreed': DataSource(
                name='Fear & Greed Index',
                url='https://api.alternative.me/fng',
                rate_limit=60
            ),
            'coinmarketcap': DataSource(
                name='CoinMarketCap',
                url='https://pro-api.coinmarketcap.com/v1',
                api_key=None,  # باید توسط کاربر تنظیم شود
                rate_limit=333,  # برای basic plan
                headers={'Accept': 'application/json'}
            ),
            'blockchain_info': DataSource(
                name='Blockchain.info',
                url='https://api.blockchain.info',
                rate_limit=300
            ),
            'binance_metrics': DataSource(
                name='Binance API',
                url='https://api.binance.com/api/v3',
                rate_limit=1200
            )
        }
        
        # آمار عملکرد
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limits_hit': 0,
            'avg_response_time': 0.0,
            'last_successful_request': None,
            'requests_by_source': {},
            'errors_by_source': {}
        }
        
        # Cache سیستم
        self.cache = {}
        self.cache_durations = {
            'fear_greed': timedelta(hours=1),
            'market_cap': timedelta(minutes=30),
            'news': timedelta(minutes=15),
            'social_sentiment': timedelta(minutes=10),
            'on_chain': timedelta(hours=2),
            'exchange_info': timedelta(hours=6)
        }
        
        # Rate limiting
        self.rate_limiters = {}
        for source_name in self.data_sources:
            self.rate_limiters[source_name] = []

    async def get_fear_greed_index(self) -> Optional[Dict]:
        """دریافت شاخص Fear & Greed"""
        cache_key = 'fear_greed'
        
        # بررسی cache
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            url = f"{self.data_sources['feargreed'].url}/?limit=30&format=json"
            
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # پردازش داده‌ها
                        if 'data' in data and len(data['data']) > 0:
                            current = data['data'][0]
                            
                            result = {
                                'current_index': int(current['value']),
                                'classification': current['value_classification'],
                                'timestamp': current['timestamp'],
                                'historical': data['data'][:7],  # آخرین 7 روز
                                'trend': self._calculate_fear_greed_trend(data['data'])
                            }
                            
                            # ذخیره در cache
                            self._cache_data(cache_key, result)
                            self._update_success_stats(time.time() - start_time, 'feargreed')
                            
                            return result
                    
                    print(f"❌ Fear & Greed API error: {response.status}")
                    return None
                    
        except Exception as e:
            print(f"❌ خطا در دریافت Fear & Greed: {e}")
            self._update_error_stats('feargreed', str(e))
            return None

    async def get_market_overview(self) -> Optional[Dict]:
        """دریافت نمای کلی بازار کریپتو"""
        cache_key = 'market_overview'
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            # داده‌های کلی بازار از CoinGecko
            url = f"{self.data_sources['coingecko'].url}/global"
            
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                
                async with session.get(url, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'data' in data:
                            global_data = data['data']
                            
                            result = {
                                'total_market_cap_usd': global_data.get('total_market_cap', {}).get('usd', 0),
                                'total_volume_24h_usd': global_data.get('total_volume', {}).get('usd', 0),
                                'market_cap_change_24h': global_data.get('market_cap_change_percentage_24h_usd', 0),
                                'bitcoin_dominance': global_data.get('market_cap_percentage', {}).get('btc', 0),
                                'ethereum_dominance': global_data.get('market_cap_percentage', {}).get('eth', 0),
                                'active_cryptocurrencies': global_data.get('active_cryptocurrencies', 0),
                                'markets': global_data.get('markets', 0),
                                'updated_at': global_data.get('updated_at')
                            }
                            
                            self._cache_data(cache_key, result)
                            self._update_success_stats(time.time() - start_time, 'coingecko')
                            
                            return result
                    
                    print(f"❌ CoinGecko market overview error: {response.status}")
                    return None
                    
        except Exception as e:
            print(f"❌ خطا در دریافت نمای کلی بازار: {e}")
            self._update_error_stats('coingecko', str(e))
            return None

    async def get_bitcoin_metrics(self) -> Optional[Dict]:
        """دریافت متریک‌های آن‌چین بیت‌کوین"""
        cache_key = 'bitcoin_onchain'
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            # معیارهای ساده از blockchain.info
            metrics = {}
            
            # آمار کلی شبکه
            async with aiohttp.ClientSession() as session:
                # هش ریت
                try:
                    async with session.get(
                        'https://api.blockchain.info/stats',
                        timeout=10
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            metrics.update({
                                'hash_rate': data.get('hash_rate', 0),
                                'difficulty': data.get('difficulty', 0),
                                'blocks_size': data.get('blocks_size', 0),
                                'total_btc': data.get('totalbc', 0) / 100000000,  # تبدیل از satoshi
                                'unconfirmed_transactions': data.get('n_tx', 0)
                            })
                except:
                    pass
                
                # mempool size
                try:
                    async with session.get(
                        'https://api.blockchain.info/q/unconfirmedcount',
                        timeout=10
                    ) as response:
                        if response.status == 200:
                            unconfirmed = await response.text()
                            metrics['mempool_size'] = int(unconfirmed)
                except:
                    pass
            
            if metrics:
                self._cache_data(cache_key, metrics)
                return metrics
            
            return None
            
        except Exception as e:
            print(f"❌ خطا در دریافت Bitcoin metrics: {e}")
            return None

    async def get_trending_coins(self, limit: int = 10) -> Optional[List[Dict]]:
        """دریافت کوین‌های ترند"""
        cache_key = 'trending_coins'
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            url = f"{self.data_sources['coingecko'].url}/search/trending"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'coins' in data:
                            trending = []
                            for coin_data in data['coins'][:limit]:
                                coin = coin_data['item']
                                trending.append({
                                    'id': coin['id'],
                                    'name': coin['name'],
                                    'symbol': coin['symbol'],
                                    'market_cap_rank': coin.get('market_cap_rank'),
                                    'price_btc': coin.get('price_btc'),
                                    'score': coin.get('score', 0)
                                })
                            
                            self._cache_data(cache_key, trending)
                            return trending
            
            return None
            
        except Exception as e:
            print(f"❌ خطا در دریافت trending coins: {e}")
            return None

    async def get_exchange_volumes(self) -> Optional[Dict]:
        """دریافت حجم معاملات صرافی‌ها"""
        cache_key = 'exchange_volumes'
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            url = f"{self.data_sources['coingecko'].url}/exchanges"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        exchanges = {}
                        for exchange in data[:20]:  # top 20 exchanges
                            exchanges[exchange['id']] = {
                                'name': exchange['name'],
                                'volume_24h_btc': exchange.get('trade_volume_24h_btc', 0),
                                'trust_score': exchange.get('trust_score', 0),
                                'country': exchange.get('country'),
                                'year_established': exchange.get('year_established')
                            }
                        
                        self._cache_data(cache_key, exchanges)
                        return exchanges
            
            return None
            
        except Exception as e:
            print(f"❌ خطا در دریافت exchange volumes: {e}")
            return None

    async def get_market_sentiment_summary(self) -> Optional[Dict]:
        """دریافت خلاصه sentiment بازار"""
        try:
            # ترکیب چندین منبع sentiment
            fear_greed = await self.get_fear_greed_index()
            market_overview = await self.get_market_overview()
            bitcoin_metrics = await self.get_bitcoin_metrics()
            trending = await self.get_trending_coins(5)
            
            # محاسبه sentiment کلی
            sentiment_score = 50  # neutral base
            sentiment_factors = []
            
            if fear_greed:
                fg_score = fear_greed['current_index']
                sentiment_score = (sentiment_score + fg_score) / 2
                sentiment_factors.append(f"Fear&Greed: {fg_score} ({fear_greed['classification']})")
            
            if market_overview:
                market_change = market_overview.get('market_cap_change_24h', 0)
                if market_change > 5:
                    sentiment_score += 10
                elif market_change < -5:
                    sentiment_score -= 10
                sentiment_factors.append(f"Market Cap 24h: {market_change:+.1f}%")
            
            # تعیین sentiment کلی
            if sentiment_score >= 75:
                overall_sentiment = "EXTREME_GREED"
            elif sentiment_score >= 55:
                overall_sentiment = "GREED"
            elif sentiment_score >= 45:
                overall_sentiment = "NEUTRAL"
            elif sentiment_score >= 25:
                overall_sentiment = "FEAR"
            else:
                overall_sentiment = "EXTREME_FEAR"
            
            return {
                'overall_sentiment': overall_sentiment,
                'sentiment_score': round(sentiment_score, 1),
                'factors': sentiment_factors,
                'fear_greed_data': fear_greed,
                'market_overview': market_overview,
                'bitcoin_metrics': bitcoin_metrics,
                'trending_coins': trending,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ خطا در محاسبه market sentiment: {e}")
            return None

    async def get_symbol_fundamentals(self, symbol: str) -> Optional[Dict]:
        """دریافت اطلاعات بنیادی یک نماد"""
        try:
            # تبدیل symbol به coin id
            symbol_lower = symbol.replace('/USDT', '').lower()
            
            # mapping معمول symbols
            coin_mapping = {
                'btc': 'bitcoin',
                'eth': 'ethereum',
                'sol': 'solana',
                'ada': 'cardano',
                'dot': 'polkadot',
                'link': 'chainlink',
                'matic': 'polygon',
                'avax': 'avalanche-2'
            }
            
            coin_id = coin_mapping.get(symbol_lower, symbol_lower)
            
            url = f"{self.data_sources['coingecko'].url}/coins/{coin_id}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        market_data = data.get('market_data', {})
                        
                        return {
                            'name': data.get('name'),
                            'symbol': data.get('symbol', '').upper(),
                            'current_price': market_data.get('current_price', {}).get('usd', 0),
                            'market_cap': market_data.get('market_cap', {}).get('usd', 0),
                            'market_cap_rank': market_data.get('market_cap_rank'),
                            'volume_24h': market_data.get('total_volume', {}).get('usd', 0),
                            'price_change_24h': market_data.get('price_change_percentage_24h', 0),
                            'price_change_7d': market_data.get('price_change_percentage_7d', 0),
                            'price_change_30d': market_data.get('price_change_percentage_30d', 0),
                            'ath': market_data.get('ath', {}).get('usd', 0),
                            'ath_change_percentage': market_data.get('ath_change_percentage', {}).get('usd', 0),
                            'atl': market_data.get('atl', {}).get('usd', 0),
                            'circulating_supply': market_data.get('circulating_supply', 0),
                            'total_supply': market_data.get('total_supply', 0),
                            'max_supply': market_data.get('max_supply'),
                            'last_updated': market_data.get('last_updated')
                        }
            
            return None
            
        except Exception as e:
            print(f"❌ خطا در دریافت fundamentals {symbol}: {e}")
            return None

    def _calculate_fear_greed_trend(self, historical_data: List[Dict]) -> str:
        """محاسبه روند Fear & Greed Index"""
        if len(historical_data) < 3:
            return 'INSUFFICIENT_DATA'
        
        values = [int(item['value']) for item in historical_data[:7]]
        
        # محاسبه میانگین متحرک
        recent_avg = sum(values[:3]) / 3
        older_avg = sum(values[3:6]) / 3 if len(values) >= 6 else recent_avg
        
        if recent_avg > older_avg + 5:
            return 'IMPROVING'
        elif recent_avg < older_avg - 5:
            return 'DETERIORATING'
        else:
            return 'STABLE'

    def _is_cache_valid(self, cache_key: str) -> bool:
        """بررسی اعتبار cache"""
        if cache_key not in self.cache:
            return False
        
        cache_entry = self.cache[cache_key]
        duration = self.cache_durations.get(cache_key, timedelta(minutes=30))
        
        return datetime.now() - cache_entry['timestamp'] < duration

    def _cache_data(self, cache_key: str, data: Any):
        """ذخیره داده در cache"""
        with self.lock:
            self.cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now()
            }

    def _update_success_stats(self, response_time: float, source: str):
        """به‌روزرسانی آمار موفقیت"""
        with self.lock:
            self.stats['total_requests'] += 1
            self.stats['successful_requests'] += 1
            self.stats['last_successful_request'] = datetime.now()
            
            # آمار per source
            if source not in self.stats['requests_by_source']:
                self.stats['requests_by_source'][source] = 0
            self.stats['requests_by_source'][source] += 1
            
            # میانگین response time
            total_successful = self.stats['successful_requests']
            current_avg = self.stats['avg_response_time']
            self.stats['avg_response_time'] = (
                (current_avg * (total_successful - 1) + response_time) / total_successful
            )

    def _update_error_stats(self, source: str, error: str):
        """به‌روزرسانی آمار خطا"""
        with self.lock:
            self.stats['total_requests'] += 1
            self.stats['failed_requests'] += 1
            
            if source not in self.stats['errors_by_source']:
                self.stats['errors_by_source'][source] = []
            
            self.stats['errors_by_source'][source].append({
                'error': error,
                'timestamp': datetime.now()
            })
            
            # حذف خطاهای قدیمی (بیش از 24 ساعت)
            cutoff = datetime.now() - timedelta(hours=24)
            self.stats['errors_by_source'][source] = [
                err for err in self.stats['errors_by_source'][source]
                if err['timestamp'] > cutoff
            ]

    def get_connector_status(self) -> Dict:
        """وضعیت کانکتور و آمار عملکرد"""
        with self.lock:
            success_rate = (
                self.stats['successful_requests'] / self.stats['total_requests'] * 100
                if self.stats['total_requests'] > 0 else 0
            )
            
            return {
                'status': 'active',
                'data_sources': {
                    name: source.active 
                    for name, source in self.data_sources.items()
                },
                'cache_entries': len(self.cache),
                'performance': {
                    'total_requests': self.stats['total_requests'],
                    'success_rate': f"{success_rate:.1f}%",
                    'avg_response_time': f"{self.stats['avg_response_time']:.2f}s",
                    'rate_limits_hit': self.stats['rate_limits_hit']
                },
                'requests_by_source': self.stats['requests_by_source'],
                'last_successful_request': self.stats['last_successful_request']
            }

    async def test_connection(self) -> bool:
        """تست اتصال به منابع داده خارجی"""
        try:
            print("🧪 تست اتصال به منابع داده خارجی...")
            
            # تست Fear & Greed
            fg_result = await self.get_fear_greed_index()
            if fg_result:
                print("✅ Fear & Greed Index: OK")
            else:
                print("❌ Fear & Greed Index: Failed")
            
            # تست Market Overview
            market_result = await self.get_market_overview()
            if market_result:
                print("✅ Market Overview: OK")
            else:
                print("❌ Market Overview: Failed")
            
            # تست Bitcoin Metrics
            btc_result = await self.get_bitcoin_metrics()
            if btc_result:
                print("✅ Bitcoin Metrics: OK")
            else:
                print("❌ Bitcoin Metrics: Failed")
            
            # حداقل 2 منبع باید کار کند
            working_sources = sum([
                1 if fg_result else 0,
                1 if market_result else 0,
                1 if btc_result else 0
            ])
            
            if working_sources >= 2:
                print(f"✅ External Data Connector: {working_sources}/3 منبع فعال")
                return True
            else:
                print(f"❌ External Data Connector: فقط {working_sources}/3 منبع فعال")
                return False
                
        except Exception as e:
            print(f"❌ خطا در تست External Data Connector: {e}")
            return False

    def clear_cache(self):
        """پاک کردن cache"""
        with self.lock:
            self.cache.clear()
            print("🗑️ Cache پاک شد")

    def __del__(self):
        """Cleanup"""
        try:
            self.clear_cache()
        except:
            pass


# تابع کمکی برای تست سریع
async def test_external_data():
    """تست سریع External Data Connector"""
    connector = ExternalDataConnector()
    
    print("=== تست External Data Connector ===")
    
    # تست Fear & Greed
    fg = await connector.get_fear_greed_index()
    if fg:
        print(f"Fear & Greed: {fg['current_index']} ({fg['classification']})")
    
    # تست Market Overview
    market = await connector.get_market_overview()
    if market:
        print(f"Total Market Cap: ${market['total_market_cap_usd']:,.0f}")
        print(f"BTC Dominance: {market['bitcoin_dominance']:.1f}%")
    
    # تست Sentiment Summary
    sentiment = await connector.get_market_sentiment_summary()
    if sentiment:
        print(f"Overall Sentiment: {sentiment['overall_sentiment']} ({sentiment['sentiment_score']})")
    
    # نمایش آمار
    status = connector.get_connector_status()
    print(f"\nStatus: {status['performance']}")

if __name__ == "__main__":
    asyncio.run(test_external_data())
