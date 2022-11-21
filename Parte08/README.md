Parte 8 - SAVI
==============
Miguel Riem Oliveira <mriem@ua.pt>
2022-2023

# Sumário

- Otimização aplicada a visão por computador
- _Image mosaicking_ e correção de côr
- Alinhamento 

# Exercícios

## Exercício 1 - Correção de côr

Recupere o último exercício da Parte 06. O objetivo era o de fazer _stitching_ de duas imagens de modo a produzir um mosaico.

![Image](docs/stitched.jpg)

O mosaico tem um bom alinhamento geométrico, visto que os objetos de ambas as imagens estão perfeitamente alinhados.

No entanto, a côr dos objetos está claramente desalinhada.

As imagens podem ser alteradas aplicando a todos os pixeis um fator de escala $s$ e um _bias_ $b$, da seguinte forma:

$I^{´}(x,y) = s \cdot I(x,y)  + b$

Implemente um script que faça a otimização destes parâmetros de modo a tornar as imagens o mais similares possível.

## Exercício 2 - Pares em imagens

Crie um script que permita clicar em pares de pontos em duas imagens, e depois grave as coordenadas desses pontos para um ficheiro json.

Em alternativa pode correr um detetor de features e fazer a associação automaticamente, como abordado na Parte06.

## Exercício 3 - alinhamento de duas imagens

Utilize as imagens da pasta **marvão** como exemplo para montar uma otimização que permita alinhar as duas imagens.

## Exercício 4 - alinhamento de múltiplas imagens

Utilize as imagens da pasta **mountain** e tente alterar o script do exercício anterior de modo a que funcione para um número variável de imagens.
