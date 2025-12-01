"""
Агент 1: Экономический Аналитик
Использует World Bank API для получения макроэкономических данных
"""

import os
import requests
from typing import Optional
from langchain_openai import ChatOpenAI
try:
    from langchain.agents import AgentExecutor, create_openai_tools_agent
except ImportError:
    from langchain.agents import AgentExecutor
    from langchain.agents import create_openai_tools_agent
from langchain.tools import tool
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

# Настройка API ключа
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# ====== HELPERS ======

def fetch_world_bank_data(country_code: str, indicator: str, start_year: int = 2020, end_year: int = 2023):
    """Получить данные из World Bank API"""
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}"
    params = {
        'format': 'json',
        'date': f'{start_year}:{end_year}',
        'per_page': 100
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if len(data) > 1 and data[1]:
            return data[1]
        return []
    except Exception as e:
        print(f"Ошибка запроса к API: {e}")
        return []


# ====== TOOLS (Инструменты) ======

@tool
def get_gdp(country_code: str, year: Optional[int] = None) -> str:
    """
    Получить ВВП (GDP) страны за определённый год.

    Args:
        country_code: Код страны (например, 'KZ' для Казахстана, 'RU' для России, 'US' для США)
        year: Год (например, 2022). Если не указан, берётся последний доступный

    Returns:
        Строка с информацией о ВВП
    """
    try:
        # NY.GDP.MKTP.CD - GDP (current US$)
        indicator = 'NY.GDP.MKTP.CD'

        if year:
            start_year = year
            end_year = year
        else:
            start_year = 2018
            end_year = 2023

        data = fetch_world_bank_data(country_code, indicator, start_year, end_year)

        if not data:
            return f"Данные о ВВП для страны {country_code} недоступны"

        # Берём первую запись (самую свежую)
        latest = data[0]
        if latest['value']:
            country_name = latest['country']['value']
            year_val = latest['date']
            value = latest['value']
            return f"ВВП {country_name} в {year_val} году: ${value:,.0f}"
        else:
            return f"Данные о ВВП для страны {country_code} за указанный период недоступны"

    except Exception as e:
        return f"Ошибка при получении данных о ВВП: {str(e)}"


@tool
def get_inflation(country_code: str, year: Optional[int] = None) -> str:
    """
    Получить уровень инфляции в стране за определённый год.

    Args:
        country_code: Код страны (например, 'KZ', 'RU', 'US')
        year: Год (например, 2022). Если не указан, берётся последний доступный

    Returns:
        Строка с информацией об инфляции
    """
    try:
        # FP.CPI.TOTL.ZG - Inflation, consumer prices (annual %)
        indicator = 'FP.CPI.TOTL.ZG'

        if year:
            start_year = year
            end_year = year
        else:
            start_year = 2018
            end_year = 2023

        data = fetch_world_bank_data(country_code, indicator, start_year, end_year)

        if not data:
            return f"Данные об инфляции для страны {country_code} недоступны"

        latest = data[0]
        if latest['value'] is not None:
            country_name = latest['country']['value']
            year_val = latest['date']
            value = latest['value']
            return f"Инфляция в {country_name} в {year_val} году: {value:.2f}%"
        else:
            return f"Данные об инфляции для страны {country_code} за указанный период недоступны"

    except Exception as e:
        return f"Ошибка при получении данных об инфляции: {str(e)}"


@tool
def get_unemployment(country_code: str, year: Optional[int] = None) -> str:
    """
    Получить уровень безработицы в стране за определённый год.

    Args:
        country_code: Код страны (например, 'KZ', 'RU', 'US')
        year: Год (например, 2022). Если не указан, берётся последний доступный

    Returns:
        Строка с информацией о безработице
    """
    try:
        # SL.UEM.TOTL.ZS - Unemployment, total (% of total labor force)
        indicator = 'SL.UEM.TOTL.ZS'

        if year:
            start_year = year
            end_year = year
        else:
            start_year = 2018
            end_year = 2023

        data = fetch_world_bank_data(country_code, indicator, start_year, end_year)

        if not data:
            return f"Данные о безработице для страны {country_code} недоступны"

        latest = data[0]
        if latest['value'] is not None:
            country_name = latest['country']['value']
            year_val = latest['date']
            value = latest['value']
            return f"Безработица в {country_name} в {year_val} году: {value:.2f}%"
        else:
            return f"Данные о безработице для страны {country_code} за указанный период недоступны"

    except Exception as e:
        return f"Ошибка при получении данных о безработице: {str(e)}"


@tool
def compare_countries(country1_code: str, country2_code: str, indicator_type: str = "gdp") -> str:
    """
    Сравнить два государства по экономическому показателю.

    Args:
        country1_code: Код первой страны (например, 'KZ')
        country2_code: Код второй страны (например, 'UZ')
        indicator_type: Тип показателя - 'gdp', 'inflation', или 'unemployment'

    Returns:
        Строка со сравнением стран
    """
    try:
        indicators = {
            'gdp': 'NY.GDP.MKTP.CD',
            'inflation': 'FP.CPI.TOTL.ZG',
            'unemployment': 'SL.UEM.TOTL.ZS'
        }

        indicator_names = {
            'gdp': 'ВВП',
            'inflation': 'Инфляция',
            'unemployment': 'Безработица'
        }

        if indicator_type not in indicators:
            return f"Неизвестный тип показателя: {indicator_type}. Используйте: gdp, inflation, unemployment"

        indicator = indicators[indicator_type]
        indicator_name = indicator_names[indicator_type]

        # Получаем данные для обеих стран
        data1 = fetch_world_bank_data(country1_code, indicator, 2020, 2023)
        data2 = fetch_world_bank_data(country2_code, indicator, 2020, 2023)

        if not data1 or not data2:
            return "Данные для сравнения недоступны"

        # Берём последнюю запись
        latest1 = data1[0]
        latest2 = data2[0]

        if latest1['value'] is None or latest2['value'] is None:
            return "Данные для одной из стран недоступны"

        country1_name = latest1['country']['value']
        country2_name = latest2['country']['value']
        year = latest1['date']
        value1 = latest1['value']
        value2 = latest2['value']

        result = f"Сравнение по показателю '{indicator_name}' за {year} год:\n\n"

        if indicator_type == 'gdp':
            result += f"- {country1_name}: ${value1:,.0f}\n"
            result += f"- {country2_name}: ${value2:,.0f}\n"
            diff = abs(value1 - value2)
            result += f"\nРазница: ${diff:,.0f}"
            if value1 > value2:
                result += f"\n{country1_name} имеет больший ВВП"
            else:
                result += f"\n{country2_name} имеет больший ВВП"
        else:
            result += f"- {country1_name}: {value1:.2f}%\n"
            result += f"- {country2_name}: {value2:.2f}%\n"
            diff = abs(value1 - value2)
            result += f"\nРазница: {diff:.2f}%"

        return result

    except Exception as e:
        return f"Ошибка при сравнении стран: {str(e)}"


@tool
def calculate_growth_rate(country_code: str, start_year: int, end_year: int) -> str:
    """
    Рассчитать темп роста ВВП страны за период.

    Args:
        country_code: Код страны (например, 'KZ')
        start_year: Начальный год
        end_year: Конечный год

    Returns:
        Строка с информацией о темпе роста
    """
    try:
        indicator = 'NY.GDP.MKTP.CD'
        data = fetch_world_bank_data(country_code, indicator, start_year, end_year)

        if not data:
            return f"Данные для расчёта темпа роста недоступны"

        # Находим значения для start_year и end_year
        start_value = None
        end_value = None
        country_name = None

        for item in data:
            if item['date'] == str(start_year) and item['value']:
                start_value = item['value']
                country_name = item['country']['value']
            if item['date'] == str(end_year) and item['value']:
                end_value = item['value']
                if not country_name:
                    country_name = item['country']['value']

        if start_value is None or end_value is None:
            return f"Данные за указанный период ({start_year}-{end_year}) недоступны"

        # Рассчитываем темп роста
        growth_rate = ((end_value - start_value) / start_value) * 100

        result = f"Темп роста ВВП {country_name} ({start_year}-{end_year}):\n\n"
        result += f"- {start_year}: ${start_value:,.0f}\n"
        result += f"- {end_year}: ${end_value:,.0f}\n"
        result += f"- Темп роста: {growth_rate:+.2f}%"

        return result

    except Exception as e:
        return f"Ошибка при расчёте темпа роста: {str(e)}"


# ====== СОЗДАНИЕ АГЕНТА ======

def create_economic_agent():
    """Создать экономического агента"""

    # Инициализация LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Список инструментов
    tools = [
        get_gdp,
        get_inflation,
        get_unemployment
    ]

    # Создание памяти (Summary Buffer Memory)
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        max_token_limit=200
    )

    # Системный промпт
    system_prompt = """Ты - экономический аналитик-эксперт, использующий данные Всемирного Банка.

Твои возможности:
- Анализировать макроэкономические показатели стран (ВВП, инфляция, безработица)
- Сравнивать экономики разных государств
- Рассчитывать темпы роста экономики
- Давать аналитические выводы на основе данных

Коды стран (примеры):
- KZ - Казахстан
- RU - Россия
- US - США
- CN - Китай
- UZ - Узбекистан
- DE - Германия
- GB - Великобритания
- FR - Франция

Всегда используй доступные инструменты для получения актуальных данных.
Давай краткие и понятные ответы с конкретными цифрами.
Запоминай, какие страны интересуют пользователя."""

    # Создание промпта
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # Создание агента
    agent = create_openai_tools_agent(llm, tools, prompt)

    # Создание executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent_executor


# ====== ИНТЕРАКТИВНЫЙ РЕЖИМ ======

def interactive_mode():
    """Интерактивный режим общения с агентом"""
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 60)
    print("Экономический Аналитик")
    print("Данные: World Bank API")
    print("=" * 60)
    print("Команды: 'exit' или 'quit' для выхода\n")

    agent = create_economic_agent()

    while True:
        try:
            user_input = input("Вы: ").strip()

            if user_input.lower() in ['exit', 'quit', 'выход']:
                print("\nДо свидания!")
                break

            if not user_input:
                continue

            print("\nАгент:")
            response = agent.invoke({"input": user_input})
            print(f"{response['output']}\n")

        except KeyboardInterrupt:
            print("\n\nДо свидания!")
            break
        except Exception as e:
            print(f"\nОшибка: {e}\n")


if __name__ == "__main__":
    # Установка зависимостей:
    # pip install langchain langchain-openai requests

    # Проверка наличия API ключа
    if not os.getenv("OPENAI_API_KEY"):
        print("ВНИМАНИЕ: Установите переменную окружения OPENAI_API_KEY")
        print("Пример: set OPENAI_API_KEY=your-key-here\n")

    # Запуск в интерактивном режиме
    interactive_mode()
