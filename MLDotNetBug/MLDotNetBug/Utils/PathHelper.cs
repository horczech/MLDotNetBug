namespace MLDotNetBug.Utils;

public static class PathHelper
{
    public static string GetAbsolutePath(string relativePath)
    {
        var dataRoot = new FileInfo(typeof(PathHelper).Assembly.Location);
        var assemblyFolderPath = dataRoot.Directory.FullName;

        var fullPath = Path.Combine(assemblyFolderPath, relativePath);

        return fullPath;
    }
}