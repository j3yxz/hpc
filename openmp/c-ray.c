/* */
/******************************************************************************
 * c-ray - a simple raytracer
 *
 * http://www.futuretech.blinkenlights.nl/c-ray.html
 *
 * Copyright (C) 2006 John Tsiombikas <nuclear@siggraph.org>
 * Copyright (C) 2016, 2017, 2018, 2020 Moreno Marzolla <moreno.marzolla@unibo.it>
 *
 * You are free to use, modify and redistribute this program under the
 * terms of the GNU General Public License v2 or (at your option) later.
 * see "http://www.gnu.org/licenses/gpl.txt" for details.
 * ---------------------------------------------------------------------------
 * Usage:
 *   compile:  gcc -Wall -Wpedantic -fopenmp -O2 -o c-ray c-ray.c -lm
 *   run:      ./c-ray -s 1280x1024 < sphfract.small.in > sphfract.ppm
 *   convert:  convert sphfract.ppm sphfract.jpeg
 * ---------------------------------------------------------------------------
 * Scene file format:
 *   # sphere (many)
 *   s  x y z  rad   r g b   shininess   reflectivity
 *   # light (many)
 *   l  x y z
 *   # camera (one)
 *   c  x y z  fov   tx ty tz
 ******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <math.h>
#include <ctype.h> 
#include <errno.h>
#include <stdint.h> /* for uint8_t */
#include <time.h>
#include <sys/time.h>
#include <assert.h>

typedef struct {
    double x, y, z;
} vec3_t;

typedef struct {
    vec3_t orig, dir;
} ray_t;

typedef struct {
    vec3_t col;         /* color */
    double spow;	/* specular power */
    double refl;	/* reflection intensity */
} material_t;

typedef struct sphere {
    vec3_t pos;
    double rad;
    material_t mat;
    struct sphere *next;
} sphere_t;

typedef struct {
    vec3_t pos, normal, vref;	/* position, normal and view reflection */
    double dist;		/* parametric distance of intersection along the ray */
} spoint_t;

typedef struct {
    vec3_t pos, targ;
    double fov;
} camera_t;

typedef struct {
    uint8_t r;  /* red   */
    uint8_t g;  /* green */
    uint8_t b;  /* blue  */
} pixel_t;

/* forward declarations */
vec3_t trace(ray_t ray, int depth);
vec3_t shade(sphere_t *obj, spoint_t *sp, int depth);

#define MAX_LIGHTS	16		/* maximum number of lights     */
#define RAY_MAG		1000.0		/* trace rays of this magnitude */
#define MAX_RAY_DEPTH	5		/* raytrace recursion limit     */
#define FOV		0.78539816	/* field of view in rads (pi/4) */
#define HALF_FOV	(FOV * 0.5)
#define ERR_MARGIN	1e-6		/* an arbitrary error margin to avoid surface acne */

/* global state */
int xres = 800;    merge(v, i, m, j, tmp);
        /* copy the sorted data back to v */
    
int yres = 600;
double aspect = 1.333333;
sphere_t *obj_list = 0;
vec3_t lights[MAX_LIGHTS];
int lnum = 0; /* number of lights */
camera_t cam;

#define NRAN	1024
#define MASK	(NRAN - 1)
vec3_t urand[NRAN];
int irand[NRAN];

const char *usage = {
    "Usage: c-ray [options]\n"
    "  Reads a scene file from stdin, writes the image to stdout, and stats to stderr.\n\n"
    "Options:\n"
    "  -s WxH     where W is the width and H the height of the image (default 800x600)\n"
    "  -r <rays>  shoot <rays> rays per pixel (antialiasing, default 1)\n"
    "  -i <file>  read from <file> instead of stdin\n"
    "  -o <file>  write to <file> instead of stdout\n"
    "  -h         this help screen\n\n"
};


/* vector dot product */
double dot(vec3_t a, vec3_t b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/* square of x */
double sq(double x)
{
    return x*x;
}

/* some helpful macros... */
#define NORMALIZE(a)  do {\
	double len = sqrt(dot(a, a));\
	(a).x /= len; (a).y /= len; (a).z /= len;\
} while(0);


/* calculate reflection vector */
vec3_t reflect(vec3_t v, vec3_t n) 
{
    vec3_t res;
    double d = dot(v, n);
    res.x = -(2.0 * d * n.x - v.x);
    res.y = -(2.0 * d * n.y - v.y);
    res.z = -(2.0 * d * n.z - v.z);
    return res;
}


vec3_t cross_product(vec3_t v1, vec3_t v2) 
{
    vec3_t res;
    res.x = v1.y * v2.z - v1.z * v2.y;
    res.y = v1.z * v2.x - v1.x * v2.z;
    res.z = v1.x * v2.y - v1.y * v2.x;
    return res;
}


/* jitter function taken from Graphics Gems I. */
vec3_t jitter(int x, int y, int s) 
{
    vec3_t pt;
    pt.x = urand[(x + (y << 2) + irand[(x + s) & MASK]) & MASK].x;
    pt.y = urand[(y + (x << 2) + irand[(y + s) & MASK]) & MASK].y;
    return pt;
}

/*
 * Compute ray-sphere intersection, and return {1, 0} meaning hit or
 * no hit.  Also the surface point parameters like position, normal,
 * etc are returned through the sp pointer if it is not NULL.
 */
int ray_sphere(const sphere_t *sph, ray_t ray, spoint_t *sp) 
{
    double a, b, c, d, sqrt_d, t1, t2;
    
    a = sq(ray.dir.x) + sq(ray.dir.y) + sq(ray.dir.z);
    b = 2.0 * ray.dir.x * (ray.orig.x - sph->pos.x) +
        2.0 * ray.dir.y * (ray.orig.y - sph->pos.y) +
        2.0 * ray.dir.z * (ray.orig.z - sph->pos.z);
    c = sq(sph->pos.x) + sq(sph->pos.y) + sq(sph->pos.z) +
        sq(ray.orig.x) + sq(ray.orig.y) + sq(ray.orig.z) +
        2.0 * (-sph->pos.x * ray.orig.x - sph->pos.y * ray.orig.y - sph->pos.z * ray.orig.z) - sq(sph->rad);
    
    if ((d = sq(b) - 4.0 * a * c) < 0.0) 
        return 0;
    
    sqrt_d = sqrt(d);
    t1 = (-b + sqrt_d) / (2.0 * a);
    t2 = (-b - sqrt_d) / (2.0 * a);
    
    if ((t1 < ERR_MARGIN && t2 < ERR_MARGIN) || (t1 > 1.0 && t2 > 1.0)) 
        return 0;
    
    if (sp) {
        if (t1 < ERR_MARGIN) t1 = t2;
        if (t2 < ERR_MARGIN) t2 = t1;
        sp->dist = t1 < t2 ? t1 : t2;
	
        sp->pos.x = ray.orig.x + ray.dir.x * sp->dist;
        sp->pos.y = ray.orig.y + ray.dir.y * sp->dist;
        sp->pos.z = ray.orig.z + ray.dir.z * sp->dist;
	
        sp->normal.x = (sp->pos.x - sph->pos.x) / sph->rad;
        sp->normal.y = (sp->pos.y - sph->pos.y) / sph->rad;
        sp->normal.z = (sp->pos.z - sph->pos.z) / sph->rad;
        
        sp->vref = reflect(ray.dir, sp->normal);
        NORMALIZE(sp->vref);
    }
    return 1;
}


vec3_t get_sample_pos(int x, int y, int sample) 
{
    vec3_t pt;
    static double sf = 0.0;
    
    if (sf == 0.0) {
        sf = 2.0 / (double)xres;
    }
    
    pt.x = ((double)x / (double)xres) - 0.5;
    pt.y = -(((double)y / (double)yres) - 0.65) / aspect;
    
    if (sample) {
        vec3_t jt = jitter(x, y, sample);
        pt.x += jt.x * sf;
        pt.y += jt.y * sf / aspect;
    }
    return pt;
}


/* determine the primary ray corresponding to the specified pixel (x, y) */
ray_t get_primary_ray(int x, int y, int sample) 
{
    ray_t ray;
    float m[3][3];
    vec3_t i, j = {0, 1, 0}, k, dir, orig, foo;
    
    k.x = cam.targ.x - cam.pos.x;
    k.y = cam.targ.y - cam.pos.y;
    k.z = cam.targ.z - cam.pos.z;
    NORMALIZE(k);
    
    i = cross_product(j, k);
    j = cross_product(k, i);
    m[0][0] = i.x; m[0][1] = j.x; m[0][2] = k.x;
    m[1][0] = i.y; m[1][1] = j.y; m[1][2] = k.y;
    m[2][0] = i.z; m[2][1] = j.z; m[2][2] = k.z;
    
    ray.orig.x = ray.orig.y = ray.orig.z = 0.0;
    ray.dir = get_sample_pos(x, y, sample);
    ray.dir.z = 1.0 / HALF_FOV;
    ray.dir.x *= RAY_MAG;
    ray.dir.y *= RAY_MAG;
    ray.dir.z *= RAY_MAG;
    
    dir.x = ray.dir.x + ray.orig.x;
    dir.y = ray.dir.y + ray.orig.y;
    dir.z = ray.dir.z + ray.orig.z;
    foo.x = dir.x * m[0][0] + dir.y * m[0][1] + dir.z * m[0][2];
    foo.y = dir.x * m[1][0] + dir.y * m[1][1] + dir.z * m[1][2];
    foo.z = dir.x * m[2][0] + dir.y * m[2][1] + dir.z * m[2][2];
    
    orig.x = ray.orig.x * m[0][0] + ray.orig.y * m[0][1] + ray.orig.z * m[0][2] + cam.pos.x;
    orig.y = ray.orig.x * m[1][0] + ray.orig.y * m[1][1] + ray.orig.z * m[1][2] + cam.pos.y;
    orig.z = ray.orig.x * m[2][0] + ray.orig.y * m[2][1] + ray.orig.z * m[2][2] + cam.pos.z;
    
    ray.orig = orig;
    ray.dir.x = foo.x + orig.x;
    ray.dir.y = foo.y + orig.y;
    ray.dir.z = foo.z + orig.z;
    
    return ray;
}


/* 
 * Compute direct illumination with the phong reflectance model.  Also
 * handles reflections by calling trace again, if necessary.
 */
vec3_t shade(sphere_t *obj, spoint_t *sp, int depth) 
{
    int i;
    vec3_t col = {0, 0, 0};
    
    /* for all lights ... */
    for (i=0; i<lnum; i++) {
        double ispec, idiff;
        vec3_t ldir;
        ray_t shadow_ray;
        sphere_t *iter = obj_list;
        int in_shadow = 0;
        
        ldir.x = lights[i].x - sp->pos.x;
        ldir.y = lights[i].y - sp->pos.y;
        ldir.z = lights[i].z - sp->pos.z;
        
        shadow_ray.orig = sp->pos;
        shadow_ray.dir = ldir;
        
        /* shoot shadow rays to determine if we have a line of sight
           with the light */
        while (iter) {
            if (ray_sphere(iter, shadow_ray, 0)) {
                in_shadow = 1;
                break;
            }
            iter = iter->next;
        }
        
        /* and if we're not in shadow, calculate direct illumination
           with the phong model. */
        if (!in_shadow) {
            NORMALIZE(ldir);
            
            idiff = fmax(dot(sp->normal, ldir), 0.0);
            ispec = obj->mat.spow > 0.0 ? pow(fmax(dot(sp->vref, ldir), 0.0), obj->mat.spow) : 0.0;
            
            col.x += idiff * obj->mat.col.x + ispec;
            col.y += idiff * obj->mat.col.y + ispec;
            col.z += idiff * obj->mat.col.z + ispec;
        }
    }
    
    /* Also, if the object is reflective, spawn a reflection ray, and
       call trace() to calculate the light arriving from the mirror
       direction. */
    if (obj->mat.refl > 0.0) {
        ray_t ray;
        vec3_t rcol;
        
        ray.orig = sp->pos;
        ray.dir = sp->vref;
        ray.dir.x *= RAY_MAG;
        ray.dir.y *= RAY_MAG;
        ray.dir.z *= RAY_MAG;
        
        rcol = trace(ray, depth + 1);
        col.x += rcol.x * obj->mat.refl;
        col.y += rcol.y * obj->mat.refl;
        col.z += rcol.z * obj->mat.refl;
    }
    
    return col;
}


/*
 * trace a ray throught the scene recursively (the recursion happens
 * through shade() to calculate reflection rays if necessary).
 */
vec3_t trace(ray_t ray, int depth) 
{
    vec3_t col;
    spoint_t sp, nearest_sp;
    sphere_t *nearest_obj = 0;
    sphere_t *iter = obj_list;
    
    nearest_sp.dist = INFINITY;

    /* if we reached the recursion limit, bail out */
    if (depth >= MAX_RAY_DEPTH) {
        col.x = col.y = col.z = 0.0;
        return col;
    }
    
    /* find the nearest intersection ... */
    while (iter) {
        if (ray_sphere(iter, ray, &sp)) {
            if (!nearest_obj || sp.dist < nearest_sp.dist) {
                nearest_obj = iter;
                nearest_sp = sp;
            }
        }
        iter = iter->next;
    }
    
    /* and perform shading calculations as needed by calling shade() */
    if (nearest_obj) {
        col = shade(nearest_obj, &nearest_sp, depth);
    } else {
        col.x = col.y = col.z = 0.0;
    }
    
    return col;
}


/* render a frame of xsz/ysz dimensions into the provided framebuffer */
void render(int xsz, int ysz, pixel_t *fb, int samples) 
{
    int i, j;
    const double rcp_samples = 1.0 / (double)samples;
    
    /* 
     * for each subpixel, trace a ray through the scene, accumulate
     * the colors of the subpixels of each pixel, then put the colors
     * into the framebuffer.
     */

    const int n_threads = omp_get_num_threads();

#if __GNUC__ < 9
 #pragma omp parallel for num_threads(n_threads) default(none) shared(ysz,xsz,samples,i,j,fb) collapse(2)
#else
 #pragma omp parallel for num_threads(n_threads) default(none) shared(n_threads,rcp_samples,ysz,xsz,samples,i,j,fb) collapse(2)
#endif
    for (j=0; j<ysz; j++) {
        for (i=0; i<xsz; i++) {
            double r, g, b;
            int s;
            r = g = b = 0.0;
            
            for (s=0; s<samples; s++) {
                vec3_t col = trace(get_primary_ray(i, j, s), 0);
                r += col.x;
                g += col.y;
                b += col.z;
            }
            
            r = r * rcp_samples;
            g = g * rcp_samples;
            b = b * rcp_samples;
            
            fb[j*xsz+i].r = (fmin(r, 1.0) * 255.0);
            fb[j*xsz+i].g = (fmin(g, 1.0) * 255.0);
            fb[j*xsz+i].b = (fmin(b, 1.0) * 255.0);
        }
    }
}


/* Load the scene from an extremely simple scene description file */
void load_scene(FILE *fp) 
{
    char line[256], *ptr;

    obj_list = 0;
    
    while ((ptr = fgets(line, sizeof(line), fp))) {
        int nread;
        sphere_t *sph;
        char type;	

        while (*ptr == ' ' || *ptr == '\t') /* checking '\0' is implied */
            ptr++;
        if (*ptr == '#' || *ptr == '\n') 
            continue;
        
        type = *ptr; 
        ptr++;

        switch (type) {
        case 's': /* sphere */
            sph = malloc(sizeof *sph); assert(sph);
            sph->next = obj_list;
            obj_list = sph;

            nread = sscanf(ptr, "%lf %lf %lf %lf %lf %lf %lf %lf %lf",
                           &(sph->pos.x), &(sph->pos.y), &(sph->pos.z), 
                           &(sph->rad), 
                           &(sph->mat.col.x), &(sph->mat.col.y), &(sph->mat.col.z),
                           &(sph->mat.spow), &(sph->mat.refl));           
            assert(9 == nread);
            break;
        case 'l': /* light */
            nread = sscanf(ptr, "%lf %lf %lf",
                           &(lights[lnum].x), 
                           &(lights[lnum].y), 
                           &(lights[lnum].z));
            assert(3 == nread);

            lnum++;
            break;
        case 'c': /* camera */
            nread = sscanf(ptr, "%lf %lf %lf %lf %lf %lf %lf",
                           &cam.pos.x, &cam.pos.y, &cam.pos.z, 
                           &cam.fov,
                           &cam.targ.x, &cam.targ.y, &cam.targ.z);
            assert(7 == nread);
            break;
        default:
            fprintf(stderr, "unknown type: %c\n", type);
        }	
    }
}


int main(int argc, char *argv[]) 
{
    int i;
    double tstart, elapsed;
    pixel_t *pixels; /* framebuffer (where the image is drawn) */
    int rays_per_pixel = 1;
    FILE *infile = stdin, *outfile = stdout;
    
    for (i=1; i<argc; i++) {
        if (argv[i][0] == '-' && argv[i][2] == 0) {
            char *sep;
            switch(argv[i][1]) {
            case 's':
                if (!isdigit(argv[++i][0]) || !(sep = strchr(argv[i], 'x')) || !isdigit(*(sep + 1))) {
                    fputs("-s must be followed by something like \"640x480\"\n", stderr);
                    return EXIT_FAILURE;
                }
                xres = atoi(argv[i]);
                yres = atoi(sep + 1);
                aspect = (double)xres / (double)yres;
                break;
                
            case 'i':
                if (!(infile = fopen(argv[++i], "r"))) {
                    fprintf(stderr, "failed to open input file %s: %s\n", argv[i], strerror(errno));
                    return EXIT_FAILURE;
                }
                break;
                
            case 'o':
                if (!(outfile = fopen(argv[++i], "w"))) {
                    fprintf(stderr, "failed to open output file %s: %s\n", argv[i], strerror(errno));
                    return EXIT_FAILURE;
                }
                break;
                
            case 'r':
                if (!isdigit(argv[++i][0])) {
                    fputs("-r must be followed by a number (rays per pixel)\n", stderr);
                    return EXIT_FAILURE;
                }
                rays_per_pixel = atoi(argv[i]);
                break;
                
            case 'h':
                fputs(usage, stdout);
                return EXIT_SUCCESS;
		
            default:
                fprintf(stderr, "unrecognized argument: %s\n", argv[i]);
                fputs(usage, stderr);
                return EXIT_FAILURE;
            }
        } else {
            fprintf(stderr, "unrecognized argument: %s\n", argv[i]);
            fputs(usage, stderr);
            return EXIT_FAILURE;
        }
    }
    
    if (!(pixels = malloc(xres * yres * sizeof(*pixels)))) {
        perror("pixel buffer allocation failed");
        return EXIT_FAILURE;
    }
    load_scene(infile);
    
    /* initialize the random number tables for the jitter */
    for (i=0; i<NRAN; i++) urand[i].x = (double)rand() / RAND_MAX - 0.5;
    for (i=0; i<NRAN; i++) urand[i].y = (double)rand() / RAND_MAX - 0.5;
    for (i=0; i<NRAN; i++) irand[i] = (int)(NRAN * ((double)rand() / RAND_MAX));
    
    tstart = omp_get_wtime();
    render(xres, yres, pixels, rays_per_pixel);
    elapsed = omp_get_wtime() - tstart;
    
    /* output statistics to stderr */
    fprintf(stderr, "Rendering took %f seconds\n", elapsed);
    
    /* output the image */
    fprintf(outfile, "P6\n%d %d\n255\n", xres, yres);
    fwrite(pixels, sizeof(*pixels), xres*yres, outfile);
    fflush(outfile);
    
    if (infile != stdin) fclose(infile);
    if (outfile != stdout) fclose(outfile);
    return EXIT_SUCCESS;
}
