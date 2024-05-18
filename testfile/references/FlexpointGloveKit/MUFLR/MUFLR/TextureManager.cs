using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tao.DevIl;
using Tao.OpenGl;

namespace MUFLR
{
    public class TextureManager : IDisposable
    {
        Dictionary<string, Texture> _textureDatabase = new Dictionary<string, Texture>();
        public Texture Get(string textureId) { return _textureDatabase[textureId]; }
        public void LoadTexture(string TextureID, string path)
        {
            int devilID = 0;
            Il.ilGenImages(1, out devilID);
            Il.ilBindImage(devilID);
            if (!Il.ilLoadImage(path)) System.Diagnostics.Debug.Assert(false, "Could not open file, [" + path + "].");
            int width = Il.ilGetInteger(Il.IL_IMAGE_WIDTH);
            int height = Il.ilGetInteger(Il.IL_IMAGE_HEIGHT);
            int openGLID = Ilut.ilutGLBindTexImage();
            System.Diagnostics.Debug.Assert(openGLID != 0);
            Il.ilDeleteImages(1, ref devilID);
            _textureDatabase.Add(TextureID, new Texture(openGLID, width, height));
        }
        #region IDisposable Members
        public void Dispose()
        {
            foreach (Texture t in _textureDatabase.Values)
            {
                Gl.glDeleteTextures(1, new int[] { t.ID });
            }
        }
        #endregion
    }
    public struct Texture
    {
        public int ID { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public Texture(int id, int width, int height)
            : this()
        {
            ID = id;
            Width = width;
            Height = height;
        }
    }
}
