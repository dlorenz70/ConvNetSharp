using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace PngImageDemo
{
    public static class PngImageReader
    {
        public static List<PngImageEntry> Load(string filePrefixDefect, string filePrefixNoDefect, string folder)
        {
            ImageConverter imageConverter = new ImageConverter();
            var defectFiles = System.IO.Directory.EnumerateFiles(folder).Where(x => x.Contains(filePrefixDefect) && x.EndsWith("png", StringComparison.OrdinalIgnoreCase));

            var nodefectFiles = System.IO.Directory.EnumerateFiles(folder).Where(x => x.Contains(filePrefixNoDefect) && x.EndsWith("png", StringComparison.OrdinalIgnoreCase));

            var results = defectFiles.Select(f => new PngImageEntry() { FileName = f, IsDefect = true, Image = PngImageReader.ToArray(f) }).ToList();
            results.AddRange(nodefectFiles.Select(f => new PngImageEntry() { FileName = f, IsDefect = false, Image = PngImageReader.ToArray(f) }));
            return results;
        }


        private static byte[] ToArray(string imageFileName)
        {
            if (File.Exists(imageFileName))
            {
                using (Stream imageStreamSource = new FileStream(imageFileName, FileMode.Open, FileAccess.Read, FileShare.Read))
                {
                    PngBitmapDecoder decoder = new PngBitmapDecoder(imageStreamSource, BitmapCreateOptions.PreservePixelFormat, BitmapCacheOption.Default);
                    BitmapSource bitmapSource = decoder.Frames[0];
                    var result = new byte[bitmapSource.PixelWidth * bitmapSource.PixelHeight];
                    bitmapSource.CopyPixels(result, bitmapSource.PixelWidth, 0);
                    return result;
                }
            }

            return null;
        }
    }
}
