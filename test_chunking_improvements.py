"""
Script de teste para verificar as melhorias no chunking.

Testa:
1. Chunks menores (300-500 tokens)
2. Overlap pequeno (50 tokens)
3. Semantic chunking (divisão por parágrafos/seções)
4. Limpeza de boilerplate
5. Metadata enriquecida
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag.chunking import ChunkConfig, chunk_document, count_tokens


def test_chunk_sizes():
    """Testa se os chunks estão no tamanho ideal (300-500 tokens)."""
    print("\n" + "=" * 70)
    print("TESTE 1: Tamanho dos chunks (300-500 tokens)")
    print("=" * 70)

    # Texto longo com parágrafos grandes para forçar divisão em sentenças
    text = """
A Revolução Industrial foi um período de grandes transformações econômicas e sociais que mudou completamente a estrutura da sociedade europeia. Iniciada na Inglaterra no século XVIII, a Revolução Industrial marcou a transição de uma economia agrária e artesanal para uma economia industrial e mecanizada. As principais características incluíram a mecanização da produção, o uso de máquinas a vapor, o surgimento das fábricas e a concentração de trabalhadores em centros urbanos. O impacto social foi profundo e duradouro, com o êxodo rural massivo e a formação de uma nova classe trabalhadora urbana que vivia em condições precárias. As condições de trabalho nas fábricas eram extremamente difíceis, com jornadas de trabalho que chegavam a 16 horas diárias, salários baixos, ausência de direitos trabalhistas e trabalho infantil generalizado. Movimentos operários surgiram gradualmente para lutar por melhores condições de trabalho, direitos trabalhistas básicos e melhores salários. Esses movimentos foram fundamentais para as conquistas sociais que vieram nas décadas seguintes.

A tecnologia foi o motor principal da Revolução Industrial, com invenções que transformaram completamente os processos produtivos. A máquina a vapor, desenvolvida por James Watt, permitiu a mecanização de diversos setores da economia, desde a indústria têxtil até o transporte ferroviário. As fábricas substituíram as oficinas artesanais, concentrando centenas de trabalhadores sob o mesmo teto e implementando uma divisão do trabalho que aumentava drasticamente a produtividade. O tear mecânico revolucionou a indústria têxtil, permitindo a produção em massa de tecidos a preços muito mais baixos. A siderurgia também passou por grandes avanços, com novos processos de produção de ferro e aço que permitiram a construção de máquinas, pontes e ferrovias. O transporte foi transformado pela locomotiva a vapor, que conectou regiões distantes e facilitou o comércio de mercadorias em escala sem precedentes.

O impacto ambiental da industrialização começou a se fazer sentir já no século XIX, com a poluição do ar e da água nas cidades industriais. As fábricas despejavam resíduos químicos nos rios, contaminando fontes de água potável e causando problemas de saúde pública. A queima de carvão para alimentar as máquinas a vapor produzia uma fumaça densa que cobria as cidades industriais, causando problemas respiratórios na população. O desmatamento acelerado para obter madeira e carvão vegetal alterou paisagens inteiras. Apesar desses problemas, a consciência ambiental era praticamente inexistente na época, e o progresso industrial era visto como um bem absoluto que justificava qualquer custo.
    """ * 3  # Repete para ter texto suficiente

    doc = {"text": text, "source": "test.txt"}
    # Usa chunk_size=1200 para resultar em max_tokens=400 (1200/3)
    chunks = chunk_document(doc, chunk_size=1200, overlap=50)

    print(f"\nTotal de chunks gerados: {len(chunks)}")

    for i, chunk in enumerate(chunks[:5]):  # Mostra os primeiros 5
        tokens = count_tokens(chunk["text"])
        chars = len(chunk["text"])
        print(f"\nChunk {i+1}:")
        print(f"  Tokens: {tokens} (ideal: 300-500)")
        print(f"  Chars: {chars}")
        print(f"  Preview: {chunk['text'][:100]}...")

        # Verifica se está no range ideal
        if 200 <= tokens <= 500:
            print(f"  ✅ Tamanho OK")
        else:
            print(f"  ⚠️  Fora do ideal")


def test_overlap():
    """Testa se o overlap está funcionando (50 tokens)."""
    print("\n" + "=" * 70)
    print("TESTE 2: Overlap entre chunks (50 tokens)")
    print("=" * 70)

    text = """
    Primeira seção sobre a Revolução Industrial e suas características principais.
    A mecanização da produção foi um dos aspectos mais importantes deste período.
    
    Segunda seção sobre o impacto social da industrialização no século XIX.
    O êxodo rural transformou a estrutura demográfica das cidades europeias.
    
    Terceira seção sobre os movimentos operários e suas reivindicações históricas.
    As greves e manifestações marcaram a luta por direitos trabalhistas básicos.
    """ * 5

    doc = {"text": text, "source": "test.txt"}
    chunks = chunk_document(doc, chunk_size=300, overlap=50)

    if len(chunks) >= 2:
        # Compara chunks adjacentes
        for i in range(min(3, len(chunks) - 1)):
            chunk1_words = set(chunks[i]["text"].split())
            chunk2_words = set(chunks[i + 1]["text"].split())
            overlap_words = chunk1_words & chunk2_words

            print(f"\nChunks {i+1} e {i+2}:")
            print(f"  Palavras em comum: {len(overlap_words)}")
            print(f"  Overlap esperado: ~35-40 palavras (50 tokens)")

            if len(overlap_words) >= 20:
                print(f"  ✅ Overlap OK")
            else:
                print(f"  ⚠️  Overlap baixo")


def test_semantic_chunking():
    """Testa se a divisão semântica está preservando parágrafos."""
    print("\n" + "=" * 70)
    print("TESTE 3: Semantic chunking (divisão por parágrafos)")
    print("=" * 70)

    text = """
Primavera dos Povos

A Primavera dos Povos foi uma série de revoluções que ocorreram na Europa em 1848.

Três Dias Gloriosos

Os Três Dias Gloriosos foram uma revolução que ocorreu na França em julho de 1830.

Marianne como símbolo republicano

Marianne é a personificação da República Francesa e um símbolo de liberdade.
    """

    doc = {"text": text, "source": "test.txt"}
    chunks = chunk_document(doc, chunk_size=512, overlap=50)

    print(f"\nTotal de chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"  Section: {chunk.get('section_title', 'N/A')}")
        print(f"  Topic: {chunk.get('topic', 'N/A')}")
        print(f"  Text preview: {chunk['text'][:80]}...")

        # Verifica se preservou unidades semânticas
        if any(
            title in chunk["text"]
            for title in ["Primavera dos Povos", "Três Dias Gloriosos", "Marianne"]
        ):
            print(f"  ✅ Unidade semântica preservada")


def test_boilerplate_removal():
    """Testa se o boilerplate está sendo removido."""
    print("\n" + "=" * 70)
    print("TESTE 4: Remoção de boilerplate")
    print("=" * 70)

    text = """
© 2024 Editora Moderna Ltda.
ISBN: 978-85-16-12345-6
Todos os direitos reservados.
Reprodução proibida sem autorização prévia.
Impresso no Brasil.

Capítulo 3: A Revolução Industrial

A Revolução Industrial foi um período de grandes transformações.
Este capítulo explora os principais aspectos deste período histórico.
    """

    doc = {"text": text, "source": "test.txt"}
    chunks = chunk_document(doc, chunk_size=512, overlap=50)

    result_text = " ".join(c["text"] for c in chunks)

    print("\nTexto original contém:")
    print("  - Copyright: ©")
    print("  - ISBN")
    print("  - 'Reprodução proibida'")
    print("  - 'Editora Moderna'")

    print("\nTexto após limpeza:")
    boilerplate_found = []
    if "©" in result_text or "Copyright" in result_text:
        boilerplate_found.append("Copyright")
    if "ISBN" in result_text:
        boilerplate_found.append("ISBN")
    if "Reprodução proibida" in result_text:
        boilerplate_found.append("Reprodução proibida")
    if "Editora Moderna" in result_text:
        boilerplate_found.append("Editora Moderna")

    if boilerplate_found:
        print(f"  ⚠️  Boilerplate encontrado: {', '.join(boilerplate_found)}")
    else:
        print(f"  ✅ Boilerplate removido com sucesso")

    if "Capítulo 3" in result_text and "Revolução Industrial" in result_text:
        print(f"  ✅ Conteúdo real preservado")


def test_metadata():
    """Testa se os metadados enriquecidos estão sendo extraídos."""
    print("\n" + "=" * 70)
    print("TESTE 5: Metadata enriquecida")
    print("=" * 70)

    text = """
Capítulo 5: A Era Napoleônica

Napoleão Bonaparte e o Império Francês

Napoleão Bonaparte foi um líder militar e político francês que se tornou imperador.
Suas conquistas militares transformaram a Europa no início do século XIX.
    """

    doc = {
        "text": text,
        "source": "historia.pdf",
        "page": 42,
        "section": "Era Napoleônica",
    }
    chunks = chunk_document(doc, chunk_size=512, overlap=50)

    print(f"\nTotal de chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} metadata:")
        print(f"  source: {chunk.get('source', 'N/A')}")
        print(f"  page: {chunk.get('page', 'N/A')}")
        print(f"  section: {chunk.get('section', 'N/A')}")
        print(f"  chapter: {chunk.get('chapter', 'N/A')}")
        print(f"  topic: {chunk.get('topic', 'N/A')}")
        print(f"  section_title: {chunk.get('section_title', 'N/A')}")
        print(f"  chunk_id: {chunk.get('chunk_id', 'N/A')}")

        # Verifica campos obrigatórios
        required_fields = [
            "source",
            "page",
            "section",
            "chunk_id",
            "chapter",
            "topic",
            "section_title",
        ]
        missing = [f for f in required_fields if f not in chunk]

        if missing:
            print(f"  ⚠️  Campos faltando: {', '.join(missing)}")
        else:
            print(f"  ✅ Todos os campos presentes")


def main():
    """Executa todos os testes."""
    print("\n" + "=" * 70)
    print("TESTE DE MELHORIAS NO CHUNKING")
    print("=" * 70)

    test_chunk_sizes()
    test_overlap()
    test_semantic_chunking()
    test_boilerplate_removal()
    test_metadata()

    print("\n" + "=" * 70)
    print("TESTES CONCLUÍDOS")
    print("=" * 70)


if __name__ == "__main__":
    main()
