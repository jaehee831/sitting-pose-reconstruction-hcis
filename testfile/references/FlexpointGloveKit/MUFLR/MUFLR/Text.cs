using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MUFLR
{
    public class TextR
    {
        public List<Sprite> Characters = new List<Sprite>();
        public Color color;
        public bool Trivul = false;
        public string Repres = "";
        public string Letters = "abcdefghijklm";
        public string Letters2 = "nopqrstuvwxyz";
        public string Numbers = "1234567890";
        public string Widge = "+:!?.,)({}&";
        public int LetterS = 0;
        public Point CenterLocation;
        public TextR(string Msg, int size, double coorx, double coory, double space, int lettersize, TextureManager k, Color j)
        {
            color = j;
            Repres = Msg;
            LetterS = size;
            CenterLocation = new Point((float)coorx, (float)coory);
            for (int z = 0; z < Msg.Length; z++)
            {
                if (Msg[z].ToString() != " ")
                {
                    Sprite e = new Sprite();
                    e.Texture = k.Get("DefaultFont");
                    float x = 0f, y = 0f;
                    for (int h = 0; h < Letters.Length; h++) if (Msg[z].ToString().ToLowerInvariant() == Letters[h].ToString()) { x = h*lettersize; y = 15*lettersize/5; }
                    for (int h = 0; h < Letters2.Length; h++) if (Msg[z].ToString().ToLowerInvariant() == Letters2[h].ToString()) { x = h * lettersize; y = 10*lettersize/5; }
                        for (int h = 0; h < Numbers.Length; h++) if (Msg[z].ToString() == Numbers[h].ToString()) { x = h * lettersize; y = 5*lettersize/5; }
                    for (int h = 0; h < Widge.Length; h++) if (Msg[z].ToString() == Widge[h].ToString()) { x = h*lettersize; y = 0; }
                    Point TopLeft = new Point((float)(x+1) / (float)e.Texture.Width, (float)(y) / (float)e.Texture.Height);
                    e.SetUV(TopLeft, new Point(TopLeft.X+((float)(lettersize)/(float)e.Texture.Width),
                        TopLeft.Y+((float)(lettersize)/(float)e.Texture.Height)));
                    e.SetWidth(size);
                    e.SetHeight(size);
                    e.SetPosition(CenterLocation.X - (Msg.Length * (size+space)) / 2 + (z * (size+space)), CenterLocation.Y);
                    Characters.Add(e);
                }
            }
            foreach (Sprite e in Characters) e.SetColor(j);
        }
        public void Augment(string Msg, int size, double coorx, double coory, double space, int lettersize, TextureManager k, Color j)
        {
            Repres += "&"+Msg;
            LetterS = size;
            Point CenterLocation = new Point((float)coorx, (float)coory);
            for (int z = 0; z < Msg.Length; z++)
            {
                if (Msg[z].ToString() != " ")
                {
                    Sprite e = new Sprite();
                    e.Texture = k.Get("DefaultFont");
                    float x = 0f, y = 0f;
                    for (int h = 0; h < Letters.Length; h++) if (Msg[z].ToString().ToLowerInvariant() == Letters[h].ToString()) { x = h * lettersize; y = 15 * lettersize / 5; }
                    for (int h = 0; h < Letters2.Length; h++) if (Msg[z].ToString().ToLowerInvariant() == Letters2[h].ToString()) { x = h * lettersize; y = 10 * lettersize / 5; }
                    for (int h = 0; h < Numbers.Length; h++) if (Msg[z].ToString() == Numbers[h].ToString()) { x = h * lettersize; y = 5 * lettersize / 5; }
                    for (int h = 0; h < Widge.Length; h++) if (Msg[z].ToString() == Widge[h].ToString()) { x = h * lettersize; y = 0; }
                    Point TopLeft = new Point((float)(x + 1) / (float)e.Texture.Width, (float)(y) / (float)e.Texture.Height);
                    e.SetUV(TopLeft, new Point(TopLeft.X + ((float)(lettersize) / (float)e.Texture.Width),
                        TopLeft.Y + ((float)(lettersize - 3) / (float)e.Texture.Height)));
                    e.SetWidth(size);
                    e.SetHeight(size);
                    e.SetPosition(CenterLocation.X - (Msg.Length * (size + space)) / 2 + (z * (size + space)), CenterLocation.Y);
                    Characters.Add(e);
                }
            }
            foreach (Sprite e in Characters) e.SetColor(j);
        }
        public TextR(string Msg, int size, double coorx, double coory, double space, int lettersizeX, int lettersizeY, TextureManager k, Color j)
        {
            color = j;
            Repres = Msg;
            LetterS = size;
            CenterLocation = new Point((float)coorx, (float)coory);
            for (int z = 0; z < Msg.Length; z++)
            {
                if (Msg[z].ToString() != " ")
                {
                    Sprite e = new Sprite();
                    e.Texture = k.Get("DefaultFont");
                    float x = 0f, y = 0f;
                    for (int h = 0; h < Letters.Length; h++) if (Msg[z].ToString().ToLowerInvariant() == Letters[h].ToString()) { x = h*lettersizeX; y = 15*lettersizeY/5; }
                    for (int h = 0; h < Letters2.Length; h++) if (Msg[z].ToString().ToLowerInvariant() == Letters2[h].ToString()) { x = h * lettersizeX; y = 10*lettersizeY/5; }
                        for (int h = 0; h < Numbers.Length; h++) if (Msg[z].ToString() == Numbers[h].ToString()) { x = h * lettersizeX; y = 5*lettersizeY/5; }
                    for (int h = 0; h < Widge.Length; h++) if (Msg[z].ToString() == Widge[h].ToString()) { x = h*lettersizeX; y = 0; }
                    Point TopLeft = new Point((float)(x+1) / (float)e.Texture.Width, (float)(y) / (float)e.Texture.Height);
                    e.SetUV(TopLeft, new Point(TopLeft.X+((float)(lettersizeX)/(float)e.Texture.Width),
                        TopLeft.Y+((float)(lettersizeY-3)/(float)e.Texture.Height)));
                    e.SetWidth(size);
                    e.SetHeight(size);
                    e.SetPosition(CenterLocation.X - (Msg.Length * (size+space)) / 2 + (z * (size+space)), CenterLocation.Y);
                    Characters.Add(e);
                }
            }
            foreach (Sprite e in Characters) e.SetColor(j);
        }
        public TextR(string Msg, int size, double coorx, double coory, double space, int lettersize, TextureManager k, Color j, bool i)
        {
            if (!i)
            {
                color = j;
                Repres = Msg;
                LetterS = size;
                CenterLocation = new Point((float)coorx, (float)coory);
                for (int z = 0; z < Msg.Length; z++)
                {
                    if (Msg[z].ToString() != " ")
                    {
                        Sprite e = new Sprite();
                        e.Texture = k.Get("DefaultFont");
                        float x = 0f, y = 0f;
                        for (int h = 0; h < Letters.Length; h++) if (Msg[z].ToString().ToLowerInvariant() == Letters[h].ToString()) { x = h * lettersize; y = 15 * lettersize / 5; }
                        for (int h = 0; h < Letters2.Length; h++) if (Msg[z].ToString().ToLowerInvariant() == Letters2[h].ToString()) { x = h * lettersize; y = 10 * lettersize / 5; }
                        for (int h = 0; h < Numbers.Length; h++) if (Msg[z].ToString() == Numbers[h].ToString()) { x = h * lettersize; y = 5 * lettersize / 5; }
                        for (int h = 0; h < Widge.Length; h++) if (Msg[z].ToString() == Widge[h].ToString()) { x = h * lettersize; y = 0; }
                        Point TopLeft = new Point((float)(x + 1) / (float)e.Texture.Width, (float)(y) / (float)e.Texture.Height);
                        e.SetUV(TopLeft, new Point(TopLeft.X + ((float)(lettersize) / (float)e.Texture.Width),
                            TopLeft.Y + ((float)(lettersize - 3) / (float)e.Texture.Height)));
                        e.SetWidth(size);
                        e.SetHeight(size);
                        e.SetPosition(CenterLocation.X - (Msg.Length * (size + space)) / 2 + (z * (size + space)), CenterLocation.Y);
                        Characters.Add(e);
                    }
                }
                foreach (Sprite e in Characters) e.SetColor(j);
            }
            else
            {
                color = j;
                Repres = Msg;
                LetterS = size;
                CenterLocation = new Point((float)coorx, (float)coory);
                double BackX = coorx;
                for (int z = 0; z < Msg.Length; z++)
                {
                    if (Msg[z].ToString() != " ")
                    {
                        Sprite e = new Sprite();
                        e.Texture = k.Get("DefaultFont");
                        float x = 0f, y = 0f;
                        for (int h = 0; h < Letters.Length; h++) if (Msg[z].ToString().ToLowerInvariant() == Letters[h].ToString()) { x = h * lettersize; y = 15 * lettersize / 5; }
                        for (int h = 0; h < Letters2.Length; h++) if (Msg[z].ToString().ToLowerInvariant() == Letters2[h].ToString()) { x = h * lettersize; y = 10 * lettersize / 5; }
                        for (int h = 0; h < Numbers.Length; h++) if (Msg[z].ToString() == Numbers[h].ToString()) { x = h * lettersize; y = 5 * lettersize / 5; }
                        for (int h = 0; h < Widge.Length; h++) if (Msg[z].ToString() == Widge[h].ToString()) { x = h * lettersize; y = 0; }
                        Point TopLeft = new Point((float)(x + 1) / (float)e.Texture.Width, (float)(y) / (float)e.Texture.Height+1f);
                        e.SetUV(TopLeft, new Point(TopLeft.X + ((float)(lettersize) / (float)e.Texture.Width),
                            TopLeft.Y + ((float)(lettersize - 3) / (float)e.Texture.Height)));
                        e.SetWidth(size);
                        e.SetHeight(size);
                        e.SetPosition(BackX, CenterLocation.Y);
                        Characters.Add(e);
                        if (Msg[z].ToString().ToLowerInvariant() != "m" && Msg[z].ToString().ToLowerInvariant() != "w") BackX += e.GetWidth()*(3.3/5.0);
                        else BackX += e.GetWidth();
                    }
                    else BackX += space;
                }
                foreach (Sprite e in Characters) e.SetColor(j);
            }
        }
        public void ReColor(string palette, Color flavor)
        {
            string Index = "";
            for (int i = 0; i < Repres.Length; i++) if (Repres[i] != ' ') Index += Repres[i];
            int z = Index.IndexOf(palette);
            for (int i = z; i < z+palette.Length; i++)
            {
                Characters[i].SetColor(flavor);
            }
        }
    }
}
