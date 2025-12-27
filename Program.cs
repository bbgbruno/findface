using Microsoft.Extensions.Configuration;
using FaceAiSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

class Program
{
    static void Main(string[] args)
    {
        // Verifica argumentos
        if (args.Length < 1)
        {
            Console.WriteLine("Uso: MinhaApp <NomeDaTurmaBase>");
            return;
        }
        string turma = args[0];

        // Carregar configuração do appsettings.json
        IConfiguration config = new ConfigurationBuilder().AddJsonFile("appsettings.json", false, false)
            .Build();

        string pastaBaseTodasTurmas = config["PastaBaseAlunos"] ?? throw new Exception("PastaBaseAlunos não configurada.");
        string eventosDir = config["PastaEvento"] ?? throw new Exception("PastaEvento não configurada.");
        string outputDir = config["PastaOutput"] ?? throw new Exception("PastaOutput não configurada.");

        // Define pasta base concreta da turma
        string baseAlunosDir = Path.Combine(pastaBaseTodasTurmas, turma);

        if (!Directory.Exists(baseAlunosDir))
        {
            Console.WriteLine($"Erro: pasta da turma não encontrada: {baseAlunosDir}");
            return;
        }

        Console.WriteLine($"Versão EXE: 1.1.0");
        Console.WriteLine($"Turma base: {turma}");
        Console.WriteLine($"Pasta base alunos: {baseAlunosDir}");
        Console.WriteLine($"Pasta evento: {eventosDir}");
        Console.WriteLine($"Pasta saída: {outputDir}");
        Console.WriteLine();

        var detector = FaceAiSharpBundleFactory.CreateFaceDetectorWithLandmarks();
        var embedder = FaceAiSharpBundleFactory.CreateFaceEmbeddingsGenerator();

        var baseEmbeddings = new Dictionary<string, float[]>();

        // ======= CÁLCULO DE EMBEDDINGS MÉDIOS POR ALUNO =======
        foreach (var alunoDir in Directory.GetDirectories(baseAlunosDir))
        {
            string alunoName = Path.GetFileName(alunoDir);
            List<float[]> embList = new List<float[]>();

            foreach (var imgPath in Directory.GetFiles(alunoDir))
            {
                using var img = Image.Load<Rgb24>(imgPath);
                var faces = detector.DetectFaces(img);

                if (!faces.Any())
                {
                    Console.WriteLine($"[BASE] Nenhuma face detectada em {imgPath} — pulando.");
                    continue;
                }

                var face = faces.First();
                var rect = new SixLabors.ImageSharp.Rectangle(
                    (int)Math.Round(face.Box.X),
                    (int)Math.Round(face.Box.Y),
                    (int)Math.Round(face.Box.Width),
                    (int)Math.Round(face.Box.Height)
                );

                using var crop = img.Clone(ctx => ctx.Crop(rect));
                var emb = embedder.GenerateEmbedding(crop);
                embList.Add(emb);

                Console.WriteLine($"[BASE] Embedding extraído de {imgPath}");
            }

            if (embList.Count == 0)
            {
                Console.WriteLine($"[BASE] Nenhuma embedding válida para {alunoName}, pulando.");
                continue;
            }

            int dim = embList[0].Length;
            float[] avgEmb = new float[dim];

            foreach (var e in embList)
            {
                for (int i = 0; i < dim; i++)
                    avgEmb[i] += e[i];
            }

            for (int i = 0; i < dim; i++)
                avgEmb[i] /= embList.Count;

            baseEmbeddings[alunoName] = avgEmb;
            Console.WriteLine($"[BASE] Embedding médio gerado para {alunoName} ({embList.Count} imagens)");
        }

        if (!baseEmbeddings.Any())
        {
            Console.WriteLine("Nenhum embedding base gerado — saindo.");
            return;
        }

        // ======= PROCESSAR FOTOS DO EVENTO =======
        foreach (var eventoPath in Directory.GetFiles(eventosDir))
        {
            using var img = Image.Load<Rgb24>(eventoPath);
            var faces = detector.DetectFaces(img);

            if (!faces.Any())
            {
                Console.WriteLine($"[EVENTO] Nenhuma face detectada em {eventoPath}");
                continue;
            }

            int faceIdx = 0;

            foreach (var face in faces)
            {
                faceIdx++;
                var rect = new SixLabors.ImageSharp.Rectangle(
                    (int)Math.Round(face.Box.X),
                    (int)Math.Round(face.Box.Y),
                    (int)Math.Round(face.Box.Width),
                    (int)Math.Round(face.Box.Height)
                );

                try
                {
                    using var crop = img.Clone(ctx => ctx.Crop(rect));
                    var embedding = embedder.GenerateEmbedding(crop);

                    string bestMatch = null;
                    double bestDist = double.MaxValue;

                    foreach (var kv in baseEmbeddings)
                    {
                        double dist = CosineDistance(embedding, kv.Value);
                        if (dist < bestDist)
                        {
                            bestDist = dist;
                            bestMatch = kv.Key;
                        }
                    }

                    const double DIST_THRESHOLD = 0.50; // você pode ajustar depois
                    Console.WriteLine($"Foto {Path.GetFileName(eventoPath)} face #{faceIdx}: melhor = {bestMatch}, distância = {bestDist:F3}");

                    if (bestMatch != null && bestDist < DIST_THRESHOLD)
                    {
                        string alunoDir = Path.Combine(outputDir, bestMatch);
                        Directory.CreateDirectory(alunoDir);
                        string dest = Path.Combine(alunoDir, Path.GetFileName(eventoPath));
                        if (faces.Count > 1)
                        {
                            File.Copy(eventoPath, dest);
                            Console.WriteLine($" → Copiado para: {dest}");
                        }

                        else
                        {
                            File.Move(eventoPath, dest);
                            Console.WriteLine($" → Movido para: {dest}");
                        }
                            
                        break;
                    }
                }
                catch (Exception)
                {
                    Console.WriteLine($"Foto {Path.GetFileName(eventoPath)} — erro ao processar face #{faceIdx}");
                    continue;
                }
            }
        }

        Console.WriteLine("Processamento finalizado.");
    }

    static double CosineDistance(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na == 0 || nb == 0) return double.MaxValue;
        return 1.0 - (dot / (Math.Sqrt(na) * Math.Sqrt(nb)));
    }
}
