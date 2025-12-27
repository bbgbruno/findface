using Microsoft.Extensions.Configuration;
using FaceAiSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

class Program
{
    static void Main(string[] args)
    {
        //Verifica argumentos
        if (args.Length < 1)
        {
            Console.WriteLine("Uso: MinhaApp <NomeDaTurmaBase>");
            return;
        }
        string turma = args[0];

        // Carregar configuração do appsettings.json
        IConfiguration config = new ConfigurationBuilder().AddJsonFile("appsettings.json", optional: false, reloadOnChange: false).Build();

        string pastaBaseTodasTurmas = config["PastaBaseAlunos"]
            ?? throw new Exception("PastaBaseAlunos não configurada.");
        string eventosDir = config["PastaEvento"]
            ?? throw new Exception("PastaEvento não configurada.");
        string outputDir = config["PastaOutput"]
            ?? throw new Exception("PastaOutput não configurada.");

        // A pasta base concreta = pastaBaseTodasTurmas + nome da turma
        string baseAlunosDir = Path.Combine(pastaBaseTodasTurmas, turma);

        if (!Directory.Exists(baseAlunosDir))
        {
            Console.WriteLine($"Erro: pasta da turma não encontrada: {baseAlunosDir}");
            return;
        }

        Console.WriteLine($"Versao EXE: 1.0.0");
        Console.WriteLine($"Turma base: {turma}");
        Console.WriteLine($"Pasta base alunos: {baseAlunosDir}");
        Console.WriteLine($"Pasta evento: {eventosDir}");
        Console.WriteLine($"Pasta saída: {outputDir}");
        Console.WriteLine();

        
        var detector = FaceAiSharpBundleFactory.CreateFaceDetectorWithLandmarks();
        var embedder = FaceAiSharpBundleFactory.CreateFaceEmbeddingsGenerator();

        var baseEmbeddings = new Dictionary<string, float[]>();

        // Carrega base de alunos da turma especificada
        foreach (var imgPath in Directory.GetFiles(baseAlunosDir))
        {
            using var img = Image.Load<Rgb24>(imgPath);
            var faces = detector.DetectFaces(img);
            if (!faces.Any())
            {
                Console.WriteLine($"[BASE] Nenhuma face detectada em {imgPath}, pulando.");
                continue;
            }
            var face = faces.First();
            var rect = new SixLabors.ImageSharp.Rectangle((int)Math.Round(face.Box.X), (int)Math.Round(face.Box.Y), (int)Math.Round(face.Box.Width), (int)Math.Round(face.Box.Height));

            using var crop = img.Clone(ctx => ctx.Crop(rect));
            var embedding = embedder.GenerateEmbedding(crop);
            string nomeAluno = Path.GetFileNameWithoutExtension(imgPath);
            baseEmbeddings[nomeAluno] = embedding;
            Console.WriteLine($"[BASE] Embedding gerado para {nomeAluno}");
        }

        if (!baseEmbeddings.Any())
        {
            Console.WriteLine("Nenhum embedding base gerado — saindo.");
            return;
        }

        // Processar fotos do evento
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
                var rect = new SixLabors.ImageSharp.Rectangle((int)Math.Round(face.Box.X), (int)Math.Round(face.Box.Y), (int)Math.Round(face.Box.Width), (int)Math.Round(face.Box.Height));
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

                    const double DIST_THRESHOLD = 0.45;
                    Console.WriteLine($"Foto {Path.GetFileName(eventoPath)} face #{faceIdx}: melhor = {bestMatch}, distância = {bestDist:F3}");

                    if (bestMatch != null && bestDist < DIST_THRESHOLD)
                    {
                        string alunoDir = Path.Combine(outputDir, bestMatch);
                        Directory.CreateDirectory(alunoDir);
                        string dest = Path.Combine(alunoDir, Path.GetFileName(eventoPath));
                        File.Move(eventoPath, dest);
                        Console.WriteLine($" → Movido para: {dest}");
                        break;
                    }
                }
                catch (Exception)
                {
                    Console.WriteLine($"Foto {Path.GetFileName(eventoPath)}  um erro aconteceu — mantendo em evento ou mover manualmente.");
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
