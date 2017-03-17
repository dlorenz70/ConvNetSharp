using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PngImageDemo
{
    public class PngImageEntry
    {
        public byte[] Image { get; set; }

        public bool IsDefect { get; set; }

        public string FileName { get; set; }

        public override string ToString() => $"{this.FileName} is defect: {this.IsDefect}";
    }
}
